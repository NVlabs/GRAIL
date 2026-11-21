"""Dependency-light, offline previews of unreviewed free-wrist grasp proposals.

The viewer places the original open-hand surface samples using proposal wrist
transforms. It deliberately neither runs IK nor changes a native sequence.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np


OBJECT_POINT_LIMIT = 4096
HAND_POINT_LIMIT = 3000
PROPOSAL_LIMIT = 20000


def _points(value: np.ndarray, name: str, limit: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{name} must be a nonempty (N, 3) array")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit, dtype=np.int64)]
    return points


def _transforms(value: np.ndarray, name: str, count: int | None = None) -> np.ndarray:
    transforms = np.asarray(value, dtype=np.float64)
    shape = (4, 4) if count is None else (count, 4, 4)
    if transforms.shape != shape or not np.isfinite(transforms).all():
        raise ValueError(f"{name} must contain finite transforms with shape {shape}")
    if not np.allclose(transforms[..., 3, :], [0, 0, 0, 1], atol=2e-5, rtol=0):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotations = transforms[..., :3, :3]
    if not (
        np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=2e-4, rtol=0)
        and np.allclose(np.linalg.det(rotations), 1, atol=2e-4, rtol=0)
    ):
        raise ValueError(f"{name} must use proper rigid rotations, without scaling or reflection")
    return transforms


def _values(value: np.ndarray, name: str, count: int) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite ({count},) array")
    return values


def _preview_payload(inputs_path: Path, proposals_path: Path, report: dict) -> dict:
    """Validate preview geometry and emit a bounded, JSON-safe representation."""
    with np.load(inputs_path, allow_pickle=False) as inputs:
        # Object coordinates already contain the physical scale. Never rescale.
        point_key = "object_pc" if "object_pc" in inputs else "mesh_vertices"
        object_points = _points(inputs[point_key], point_key, OBJECT_POINT_LIMIT)
        current = _transforms(inputs["current_wrist_object"], "current_wrist_object")
        hand_points = (
            _points(inputs["hand_open_points_wrist"], "hand_open_points_wrist", HAND_POINT_LIMIT)
            if "hand_open_points_wrist" in inputs else None
        )
        input_labels = {}
        for label in ("side", "frame"):
            if label in inputs:
                scalar = np.asarray(inputs[label])
                if scalar.shape != ():
                    raise ValueError(f"{label} must be a scalar label")
                input_labels[label] = str(scalar.item())
    with np.load(proposals_path, allow_pickle=False) as proposals:
        confidence = np.asarray(proposals["confidence"], dtype=np.float64)
        if confidence.ndim != 1 or not 1 <= len(confidence) <= PROPOSAL_LIMIT:
            raise ValueError(f"confidence must contain 1 to {PROPOSAL_LIMIT} proposals")
        count = len(confidence)
        confidence = _values(confidence, "confidence", count)
        wrists = _transforms(proposals["wrist_object"], "wrist_object", count)
        _transforms(proposals["grasps_object"], "grasps_object", count)
        _transforms(proposals["wrist_world"], "wrist_world", count)
        shifts = _values(proposals["source_shift_m"], "source_shift_m", count)
        angles = _values(proposals["source_rotation_deg"], "source_rotation_deg", count)
        if np.any(shifts < 0) or np.any(angles < 0) or np.any(angles > 180 + 1e-4):
            raise ValueError("source displacement must be nonnegative and rotation must lie in [0, 180] degrees")
        # Stable confidence ranking preserves source row identity for later review.
        order = np.argsort(-confidence, kind="stable")
        candidates = [
            {
                "candidate_index": int(index),
                "confidence": float(confidence[index]),
                "wrist_object": wrists[index].tolist(),
                "source_shift_m": float(shifts[index]),
                "source_rotation_deg": float(angles[index]),
            }
            for index in order
        ]
    name = report.get("label", report.get("name", report.get("case_name", "Grasp proposals")))
    return {
        "name": str(name),
        "side": str(report.get("side", input_labels.get("side", "unspecified"))),
        "frame": str(report.get("frame", input_labels.get("frame", "unspecified"))),
        "units": "meters",
        "status": "UNREVIEWED_FREE_WRIST_PROPOSALS",
        "object_points": object_points.tolist(),
        "hand_points_wrist": hand_points.tolist() if hand_points is not None else None,
        "current_wrist_object": current.tolist(),
        "candidates": candidates,
    }


def write_preview(output_dir: Path, inputs_path: Path, proposals_path: Path, report: dict) -> Path:
    """Write a standalone ``preview.html`` without replacing an existing file.

    Inputs and proposals use meters and object-local rigid transforms. Required
    proposal fields are ``grasps_object``, ``wrist_object``, ``wrist_world``,
    ``confidence``, ``source_shift_m`` and ``source_rotation_deg``. Optional hand
    samples are actual original open-hand geometry in the native wrist frame.
    """
    target = Path(output_dir) / "preview.html"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Preview already exists: {target}")
    payload = _preview_payload(Path(inputs_path), Path(proposals_path), report)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    # A JSON script block is still parsed by HTML: closing tags must be escaped.
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    prefix, _, after_title = _HTML.partition("__TITLE__")
    middle, _, suffix = after_title.partition("__PAYLOAD__")
    page = prefix + html.escape(payload["name"], quote=True) + middle + encoded + suffix
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(page)
    return target


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — grasp proposal preview</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101419;color:#edf2f6}*{box-sizing:border-box}
body{margin:0;padding:24px;max-width:1500px;margin:auto}header{margin-bottom:18px}h1{font-size:24px;margin:6px 0}
p{line-height:1.5;color:#bcc7d0;margin:8px 0}.badge{font-size:12px;color:#dce98c;letter-spacing:.08em}
main{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:16px}.viewer,aside{border:1px solid #34414c;border-radius:12px;background:#172029}
.viewer{position:relative;min-width:0;overflow:hidden}canvas{display:block;width:100%;height:570px;touch-action:none;cursor:grab}
canvas:active{cursor:grabbing}.help{position:absolute;bottom:8px;left:14px;font-size:12px;color:#c9d1d8;pointer-events:none}
aside{padding:18px}label{display:block;color:#cbd4dc;margin:14px 0 7px}select,button{font:inherit;background:#253442;color:#f2f5f8;border:1px solid #536476;border-radius:6px;padding:8px}
select{width:100%}.buttons{display:flex;gap:8px;margin:10px 0}.buttons button{flex:1}button{cursor:pointer}button:disabled{opacity:.35;cursor:default}
dl{margin-top:20px}dt{color:#acb9c5;font-size:12px;margin-top:14px}dd{margin:4px 0;font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:16px;padding:12px 16px;border-top:1px solid #34414c;font-size:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.object{background:#e8b767}.proposal{background:#55d8dd}.current{background:#c4ccd2}.note{font-size:12px;color:#bbc4cb;border-top:1px solid #34414c;padding-top:12px}
@media(max-width:800px){body{padding:14px}main{grid-template-columns:1fr}canvas{height:420px}aside{padding:14px}}
</style></head><body>
<header><div class="badge">UNREVIEWED · FREE-WRIST HYPOTHESES</div><h1 id="name">Grasp proposals</h1>
<p id="context"></p><p>Explore proposed wrist poses with the original object and open-hand surface samples. These are not mounted robot poses, contact maps, collision results, or accepted motion.</p></header>
<main><section class="viewer" aria-label="Interactive grasp point-cloud preview"><canvas id="view" aria-label="Object and hand point-cloud preview"></canvas>
<div class="help">Drag to rotate · Scroll or pinch to zoom · Coordinates in meters</div>
<div class="legend"><span><i class="dot object"></i>Object</span><span><i class="dot proposal"></i>Proposed open hand</span><span><i class="dot current"></i>Current wrist / open hand</span></div></section>
<aside><label for="candidate">Candidate ranked by learned confidence</label><select id="candidate"></select>
<div class="buttons"><button id="previous" type="button">Previous</button><button id="next" type="button">Next</button></div>
<button id="reset" type="button">Center view</button><label><input id="baseline" type="checkbox" checked> Show current open-hand context</label>
<dl><dt>Proposal row (zero-based)</dt><dd id="row"></dd><dt>Learned confidence</dt><dd id="confidence"></dd>
<dt title="Distance from the current native wrist, in millimeters">Wrist displacement</dt><dd id="shift"></dd><dt title="Relative rotation from the current native wrist">Wrist rotation change</dt><dd id="rotation"></dd></dl>
<p class="note" id="geometry"></p><p class="note">Confidence ranks model proposals; it is not a probability of a valid robot grasp. Native arm IK, finger closure, source review and geometry checks are still required.</p>
</aside></main><noscript><p>This offline interactive point-cloud preview requires JavaScript. No data is sent to a server.</p></noscript>
<script id="preview-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('preview-data').textContent);
const byId=id=>document.getElementById(id),canvas=byId('view'),ctx=canvas.getContext('2d'),choice=byId('candidate');
byId('name').textContent=data.name;
byId('context').textContent=`${data.side} hand · Native frame ${data.frame} · ${data.candidates.length} proposals`;
byId('geometry').textContent=data.hand_points_wrist ? 'Hand geometry shows the same original open pose at each wrist proposal. Finger closure and the attached arm are not shown.' : 'Open-hand surface samples are unavailable. Wrist axes show proposal position and orientation only.';
for(let i=0;i<data.candidates.length;i++){const option=document.createElement('option');option.value=i;option.textContent=`${i+1}. Proposal ${data.candidates[i].candidate_index}`;choice.appendChild(option);}
const transform=(point,T)=>[0,1,2].map(i=>T[i][0]*point[0]+T[i][1]*point[1]+T[i][2]*point[2]+T[i][3]);
const currentHand=(data.hand_points_wrist||[]).map(p=>transform(p,data.current_wrist_object));
let proposedHand=[],yaw=-.65,pitch=.45,zoom=1,center=[0,0,0],radius=.25;
function selected(){return data.candidates[Number(choice.value)];}
function fit(){const points=data.object_points.concat(proposedHand,[transform([0,0,0],selected().wrist_object)]);const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const p of points)for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}center=lo.map((v,i)=>(v+hi[i])/2);radius=Math.max(.01,Math.hypot(...hi.map((v,i)=>v-lo[i]))/2);zoom=1;draw();}
function update(){const proposal=selected();proposedHand=(data.hand_points_wrist||[]).map(p=>transform(p,proposal.wrist_object));byId('row').textContent=proposal.candidate_index;byId('confidence').textContent=proposal.confidence.toFixed(5);byId('shift').textContent=(proposal.source_shift_m*1000).toFixed(1)+' mm';byId('rotation').textContent=proposal.source_rotation_deg.toFixed(1)+'°';byId('previous').disabled=Number(choice.value)===0;byId('next').disabled=Number(choice.value)===data.candidates.length-1;fit();}
function projected(p){const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;return[a,Math.cos(pitch)*z-Math.sin(pitch)*b,Math.sin(pitch)*z+Math.cos(pitch)*b];}
function draw(){const rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2),w=rect.width,h=rect.height;canvas.width=Math.max(1,Math.round(w*dpr));canvas.height=Math.max(1,Math.round(h*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#172029';ctx.fillRect(0,0,w,h);const scale=Math.min(w,h)*.41*zoom/radius;const screen=p=>{const v=projected(p);return[w/2+v[0]*scale,h/2-v[1]*scale,v[2]];};const points=[];for(const p of data.object_points)points.push({v:screen(p),color:'#e8b767',size:1.6});if(byId('baseline').checked)for(const p of currentHand)points.push({v:screen(p),color:'#83909b',size:1.3});for(const p of proposedHand)points.push({v:screen(p),color:'#55d8dd',size:1.8});points.sort((a,b)=>a.v[2]-b.v[2]);for(const p of points){ctx.fillStyle=p.color;ctx.fillRect(p.v[0]-p.size/2,p.v[1]-p.size/2,p.size,p.size);}function axes(T,current){const origin=screen(transform([0,0,0],T)),length=Math.max(.015,Math.min(.055,radius*.2));ctx.lineWidth=current?1.5:2.5;for(let i=0;i<3;i++){const local=[0,0,0];local[i]=length;const tip=screen(transform(local,T));ctx.strokeStyle=current?'#d0d9e0':['#f17982','#91d67a','#829efa'][i];ctx.beginPath();ctx.moveTo(origin[0],origin[1]);ctx.lineTo(tip[0],tip[1]);ctx.stroke();}ctx.fillStyle=current?'#edf2f6':'#55d8dd';ctx.beginPath();ctx.arc(origin[0],origin[1],3,0,Math.PI*2);ctx.fill();ctx.font='12px system-ui';ctx.fillText(current?'Current wrist':'Proposed wrist',origin[0]+7,origin[1]-8);}
axes(data.current_wrist_object,true);axes(selected().wrist_object,false);}
choice.addEventListener('change',update);byId('previous').onclick=()=>{choice.value=Math.max(0,Number(choice.value)-1);update();};byId('next').onclick=()=>{choice.value=Math.min(data.candidates.length-1,Number(choice.value)+1);update();};byId('baseline').onchange=draw;byId('reset').onclick=fit;window.addEventListener('resize',draw);
const pointers=new Map();canvas.addEventListener('pointerdown',event=>{canvas.setPointerCapture(event.pointerId);pointers.set(event.pointerId,[event.clientX,event.clientY]);});canvas.addEventListener('pointermove',event=>{if(!pointers.has(event.pointerId))return;const old=pointers.get(event.pointerId),now=[event.clientX,event.clientY];if(pointers.size===1){yaw+=(now[0]-old[0])*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(now[1]-old[1])*.008));}else{const other=[...pointers.entries()].find(([id])=>id!==event.pointerId)[1],before=Math.hypot(old[0]-other[0],old[1]-other[1]),after=Math.hypot(now[0]-other[0],now[1]-other[1]);if(before>1)zoom=Math.max(.1,Math.min(12,zoom*after/before));}pointers.set(event.pointerId,now);draw();});for(const kind of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(kind,event=>pointers.delete(event.pointerId));canvas.addEventListener('wheel',event=>{event.preventDefault();zoom=Math.max(.1,Math.min(12,zoom*Math.exp(-event.deltaY*.001)));draw();},{passive:false});update();
</script></body></html>
'''

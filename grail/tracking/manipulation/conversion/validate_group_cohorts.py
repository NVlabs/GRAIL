#!/usr/bin/env python3
"""Read-only verification of a built/staged object cohort tree and pinned resolver coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path

from build_group_cohorts import digest, pinned_modules, read, require


def validate(root):
    root=Path(root).resolve();receipt=read(root/'build_receipt.json');catalog=read(root/'catalog.json')
    _,resolve=pinned_modules();runtime=Path(catalog['runtime_root']);checks=0
    def local(path):
        p=root/Path(path).relative_to(runtime)
        require(p.resolve().is_relative_to(root),'Runtime reference escapes output tree')
        return p
    for entry in receipt['files']:
        p=local(entry['path']);require(p.is_file() and digest(p)==entry['sha256'],'Output file SHA mismatch: '+str(p));checks+=1
    require(digest(root/'catalog.json')==receipt['catalog']['sha256'],'Catalog SHA mismatch')
    aliases=[];motion_keys=[];mesh_shas=[]
    for group in catalog['groups']:
        binding_path=local(group['binding_path']);binding=read(binding_path)
        require(digest(binding_path)==group['binding_sha256'],'Binding SHA mismatch')
        require(binding['case_id']==group['group_id'] and binding['object_mesh_sha256']==group['object_mesh_sha256'],'Group binding identity mismatch')
        require(binding['motion_count']==group['motion_count']==len(binding['motions']),'Motion count mismatch')
        for entry in binding['training_pack']['files']:
            p=local(entry['path']);require(p.is_file() and digest(p)==entry['sha256'],'Runtime payload SHA mismatch: '+str(p));checks+=1
        path=local(group['roster_path']);doc=read(path)
        require(digest(path)==group['roster_sha256'],'Roster SHA mismatch')
        view=resolve(path,num_envs=group['num_envs'],rank=0,world_size=1,runtime_target_fps_hz=50)
        expected={m['motion_id'] for m in binding['motions']}
        require(expected==set(view['motion_aliases'])=={r['alias'] for r in doc['records']},'Training roster coverage differs')
        records={r['alias']:r for r in doc['records']}
        for record in records.values():
            for field in ['active_articulation_usd','static_context_usd','bps_path']:
                require(local(record[field]).is_file(),'Missing resolved record asset')
        for m in binding['motions']:
            path=local(m['single_env_roster_path']);d=read(path)
            require(digest(path)==m['single_env_roster_sha256'],'Evaluation roster SHA mismatch')
            ev=resolve(path,num_envs=1,rank=0,world_size=1,runtime_target_fps_hz=50)
            require(ev['motion_aliases']==[m['motion_id']] and len(d['records'])==len(d['slots'])==1,'Evaluation not exactly one requested motion')
            require(d['records'][0]==records[m['motion_id']],'Train/eval motion record mismatch')
            require(d['slots'][0]['compatible_motion_aliases']==[m['motion_id']],'Evaluation candidate contamination')
            aliases.append(m['motion_id']);motion_keys.append(m['source_motion_key']);checks+=4
        mesh_shas.append(group['object_mesh_sha256']);checks+=7
    require(len(mesh_shas)==len(set(mesh_shas)),'One mesh split into multiple runs')
    require(len(aliases)==len(set(aliases))==catalog['motion_count'],'Missing/duplicate aliases')
    require(len(motion_keys)==len(set(motion_keys))==catalog['expected_motion_count'],'Missing/duplicate source motion identities')
    require(set(aliases)==set(catalog['coverage']['aliases']) and not catalog['coverage']['missing'] and not catalog['coverage']['duplicates'],'Coverage declaration differs')
    return {'status':'passed','scope':'CPU/hash/manifest/native pinned resolver; not a GPU smoke result',
            'groups':len(mesh_shas),'motions':len(aliases),'checks':checks,'catalog_sha256':digest(root/'catalog.json'),
            'production_ready':catalog['production_ready'],'submitted':catalog['submitted']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(validate(a.root),indent=2))

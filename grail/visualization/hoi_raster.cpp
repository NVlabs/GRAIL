// CPU-only perspective rasterizer for review of unmodified native meshes.
#include <algorithm>
#include <cmath>
#include <cstdint>

struct Point { double x, y, z; };

static void triangle(const Point* p, int face, double fx, double fy, int h, int w,
                     float* depth, int32_t* ids) {
    double x[3], y[3];
    for (int i = 0; i < 3; ++i) {
        x[i] = fx * p[i].x / p[i].z + w / 2.0;
        y[i] = fy * p[i].y / p[i].z + h / 2.0;
    }
    double den = (y[1]-y[2])*(x[0]-x[2]) + (x[2]-x[1])*(y[0]-y[2]);
    if (std::abs(den) < 1e-12) return;
    // Clamp doubles before conversion so a near-plane projection cannot overflow int.
    int xmin = static_cast<int>(std::max(0.0, std::min(double(w), std::floor(*std::min_element(x,x+3)))));
    int xmax = static_cast<int>(std::max(0.0, std::min(double(w), std::ceil(*std::max_element(x,x+3)))));
    int ymin = static_cast<int>(std::max(0.0, std::min(double(h), std::floor(*std::min_element(y,y+3)))));
    int ymax = static_cast<int>(std::max(0.0, std::min(double(h), std::ceil(*std::max_element(y,y+3)))));
    for (int yy=ymin; yy<ymax; ++yy) for (int xx=xmin; xx<xmax; ++xx) {
        double u=((y[1]-y[2])*(xx+.5-x[2])+(x[2]-x[1])*(yy+.5-y[2]))/den;
        double v=((y[2]-y[0])*(xx+.5-x[2])+(x[0]-x[2])*(yy+.5-y[2]))/den;
        double t=1-u-v;
        if (std::min({u,v,t}) < -1e-12) continue;
        float z=1.0/(u/p[0].z+v/p[1].z+t/p[2].z);
        int index=yy*w+xx;
        if (z < depth[index]) { depth[index]=z; ids[index]=face; }
    }
}

extern "C" void hoi_raster(const double* vertices, const int64_t* faces, int nf,
                           double fx, double fy, int h, int w, double near,
                           float* depth, int32_t* ids) {
    std::fill(depth, depth+h*w, INFINITY);
    std::fill(ids, ids+h*w, -1);
    for (int fi=0; fi<nf; ++fi) {
        Point p[3], clipped[4];
        for (int k=0; k<3; ++k) {
            const double* v=vertices+3*faces[fi*3+k]; p[k]={v[0],v[1],v[2]};
        }
        // Clip against the near plane; keep the original face ID after triangulation.
        int count=0;
        for (int k=0; k<3; ++k) {
            Point a=p[k], b=p[(k+1)%3];
            bool inside_a=a.z>=near, inside_b=b.z>=near;
            if (inside_a) clipped[count++]=a;
            if (inside_a != inside_b) {
                double alpha=(near-a.z)/(b.z-a.z);
                clipped[count++]={a.x+alpha*(b.x-a.x),a.y+alpha*(b.y-a.y),near};
            }
        }
        for (int k=1; k+1<count; ++k) {
            Point tri[3]={clipped[0],clipped[k],clipped[k+1]};
            triangle(tri,fi,fx,fy,h,w,depth,ids);
        }
    }
}

#include <algorithm>
#include <cmath>
#include <vector>
struct Linear30 { std::vector<double> w; double b; int max_batch; };
extern "C" void* lie30_create(const double* w, double b, int max_batch) {
  if (!w || !std::isfinite(b) || max_batch < 1 || max_batch > 512) return nullptr;
  for (int j=0;j<4096;j++) if (!std::isfinite(w[j])) return nullptr;
  return new Linear30{std::vector<double>(w,w+4096),b,max_batch};
}
extern "C" int lie30_run(void* ptr, const float* x, int batch, double* out) {
  if (!ptr || !x || !out || batch < 1) return 1;
  auto& h=*static_cast<Linear30*>(ptr);
  if (batch > h.max_batch) return 1;
  for (int i=0;i<batch;i++) {
    double z=h.b;
    for (int j=0;j<4096;j++) z+=h.w[j]*double(x[i*4096+j]);
    if (!std::isfinite(z)) return 2;
    z=std::clamp(z,1.,std::sqrt(2048.));
    out[i]=std::clamp(z*z,1.,2048.);
  }
  return 0;
}
extern "C" void lie30_destroy(void* ptr) { delete static_cast<Linear30*>(ptr); }

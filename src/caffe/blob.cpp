#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include <phast.h>
#include "caffe/phast_functors.hpp"

namespace caffe {

template <>
void Blob<float>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
  }

  if (count_ > capacity_) {
    if (data_v == NULL) data_v = std::make_shared<phast::vector<float> >(count_);
    else data_v->assign(count_);
    if (diff_v == NULL) diff_v = std::make_shared<phast::vector<float> >(count_);
    else diff_v->assign(count_);

//    printf("Realloc old ptr: %p bytes\n", diff_vptr);
    data_vptr = (float*) realloc(data_vptr, count_ * sizeof(float));
    diff_vptr = (float*) realloc(diff_vptr, count_ * sizeof(float));
//    printf("Realloc new ptr: %p bytes\n", diff_vptr);

    for(int i=capacity_; i < count_; i++) {
      data_vptr[i] = 0.0;
      diff_vptr[i] = 0.0;
    }

    capacity_ = count_;

    data_vlocation = 0;
    diff_vlocation = 0;
  }
}

template <>
void Blob<float>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <>
void Blob<float>::syncDataToLocal() {
  if (data_vlocation == 2) {
    memcpy(data_vptr, data_v->get_host(), count_ * sizeof(float));
    //data_vptr = data_v->get_host();
    data_vlocation = 0;
  }
}

template <>
void Blob<float>::syncDataToRemote() {
  if (data_vlocation == 1) {
    data_v->assign(data_vptr, data_vptr + data_v->size());
    data_vlocation = 0;
  }
}

template <>
void Blob<float>::syncDiffToLocal() {
  if (diff_vlocation == 2) {
    memcpy(diff_vptr, diff_v->get_host(), count_ * sizeof(float));
    //diff_vptr = diff_v->get_host();
    diff_vlocation = 0;
  }
}

template <>
void Blob<float>::syncDiffToRemote() {
  if (diff_vlocation == 1) {
    diff_v->assign(diff_vptr, diff_vptr + diff_v->size());
    diff_vlocation = 0;
  }
}

template <>
phast::vector<float>& Blob<float>::getDataVector() {
  syncDataToRemote();
  data_vlocation = 2;
  return *data_v;
}

template <>
phast::vector<float>& Blob<float>::getDiffVector() {
  syncDiffToRemote();
  diff_vlocation = 2;
  return *diff_v;
}

template <>
phast::vector<float> Blob<float>::getDataAsVector(unsigned int n) {
  syncDataToRemote();
  data_vlocation = 2;
  phast::vector<float> vec;
  vec.set_dev(n, data_v->get_dev());
  return vec;
}

template <>
phast::vector<float> Blob<float>::getDiffAsVector(unsigned int n) {
  syncDiffToRemote();
  diff_vlocation = 2;
  phast::vector<float> vec;
  vec.set_dev(n, diff_v->get_dev());
  return vec;
}

template <>
phast::matrix<float> Blob<float>::getDataAsMatrix(unsigned int rows, unsigned int cols, bool isCollumnMajor) {
  syncDataToRemote();
  data_vlocation = 2;
  phast::matrix<float> mat;
  if (isCollumnMajor) {
    mat.set_dev(cols, rows, data_v->get_dev());
    mat.transpose();
  }
  else mat.set_dev(rows, cols, data_v->get_dev());
  return mat;
}

template <>
phast::matrix<float> Blob<float>::getDiffAsMatrix(unsigned int rows, unsigned int cols, bool isCollumnMajor) {
  syncDiffToRemote();
  diff_vlocation = 2;
  phast::matrix<float> mat;
  if (isCollumnMajor) {
    mat.set_dev(cols, rows, diff_v->get_dev());
    mat.transpose();
  }
  else mat.set_dev(rows, cols, diff_v->get_dev());
  return mat;
}

template <>
phast::cube<float> Blob<float>::getDataAsCube(unsigned int i, unsigned int j, unsigned int k) {
  syncDataToRemote();
  data_vlocation = 2;
  phast::cube<float> cube;
  cube.set_dev(i, j, k, data_v->get_dev());
  return cube;
}

template <>
phast::cube<float> Blob<float>::getDiffAsCube(unsigned int i, unsigned int j, unsigned int k) {
  syncDiffToRemote();
  diff_vlocation = 2;
  phast::cube<float> cube;
  cube.set_dev(i, j, k, diff_v->get_dev());
  return cube;
}

template <>
void Blob<float>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <>
void Blob<float>::ReshapeLike(const Blob<float>& other) {
  Reshape(other.shape());
}

template <>
Blob<float>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  data_vptr = NULL;
  diff_vptr = NULL;
  Reshape(num, channels, height, width);
}

template <>
Blob<float>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  data_vptr = NULL;
  diff_vptr = NULL;
  Reshape(shape);
}

template <>
const float* Blob<float>::cpu_data() {
  syncDataToLocal();
  return (const float*)data_vptr;
}

template <>
void Blob<float>::set_cpu_data(float* data) {
  size_t size = count_;
  data_v = std::make_shared<phast::vector<float> >();
  diff_v = std::make_shared<phast::vector<float> >();
  
  data_v->assign(data, data + size);
  diff_v->assign(size);
  data_vlocation = 2;
}

template <>
const float* Blob<float>::cpu_diff() {
  syncDiffToLocal();
  return (const float*)diff_vptr;
}

template <>
float* Blob<float>::mutable_cpu_data() {
  syncDataToLocal();
  data_vlocation = 1;
  return data_vptr;
}

template <>
float* Blob<float>::mutable_cpu_diff() {
  syncDiffToLocal();
  diff_vlocation = 1;
  return diff_vptr;
}

template <>
void Blob<float>::ShareData(Blob& other) {  
  data_v = other.data_v;
  data_vptr = other.data_vptr;
  data_vlocation = 0;
}

template <>
void Blob<float>::ShareDiff(Blob& other) {
  diff_v = other.diff_v;
  diff_vptr = other.diff_vptr;
  diff_vlocation = 0;
}

template <>
void Blob<float>::Update() {
  syncDataToRemote();
  syncDiffToRemote();
  phast::for_each(data_v->begin(), data_v->end(),
		   diff_v->begin(),
		   data_v->begin(),
		   phast::minus<float>());
  data_vlocation = 2;
}

template <>
float Blob<float>::asum_data() {
  syncDataToRemote();
  return phast::accumulate(data_v->begin(), data_v->end());
}

template <>
float Blob<float>::asum_diff() {
  syncDiffToRemote();
  return phast::accumulate(diff_v->begin(), diff_v->end());
}

template <>
float Blob<float>::sumsq_data() {
  syncDataToRemote();
  return phast::dot_product(*data_v, *data_v);
}

template <>
float Blob<float>::sumsq_diff() {
  syncDiffToRemote();
  return phast::dot_product(*diff_v, *diff_v);
}

template <>
void Blob<float>::scale_data(float scale_factor) {
  syncDataToRemote();
  phast::for_each(data_v->begin(), data_v->end(), vectorScale<float>(scale_factor));
  data_vlocation = 2;
}

template <>
void Blob<float>::scale_diff(float scale_factor) {
  syncDiffToRemote();
  phast::for_each(diff_v->begin(), diff_v->end(), vectorScale<float>(scale_factor));
  diff_vlocation = 2;
}

template <>
bool Blob<float>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <>
void Blob<float>::CopyFrom(Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  
  if (copy_diff) {
    //diff_v->assign(source.cpu_diff(), source.cpu_diff() + count_);
    source.syncDiffToRemote();
    phast::copy(source.diff_v->begin(), source.diff_v->end(), diff_v->begin());
    diff_vlocation = 2;
  }
  else {
    //data_v->assign(source.cpu_data(), source.cpu_data() + count_);
    source.syncDataToRemote();
    phast::copy(source.data_v->begin(), source.data_v->end(), data_v->begin());
    data_vlocation = 2;
  }
}
  
template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
void Blob<Dtype>::syncDataToLocal() {}

template <typename Dtype>
void Blob<Dtype>::syncDataToRemote() {}

template <typename Dtype>
void Blob<Dtype>::syncDiffToLocal() {}

template <typename Dtype>
void Blob<Dtype>::syncDiffToRemote() {}

template <typename Dtype>
phast::vector<float>& Blob<Dtype>::getDataVector() {return *data_v;}

template <typename Dtype>
phast::vector<float>& Blob<Dtype>::getDiffVector() {return *diff_v;}

template <typename Dtype>
phast::vector<float> Blob<Dtype>::getDataAsVector(unsigned int n) {
  phast::vector<float> vec;
  return vec;
}

template <typename Dtype>
phast::vector<float> Blob<Dtype>::getDiffAsVector(unsigned int n) {
  phast::vector<float> vec;
  return vec;
}

template <typename Dtype>
phast::matrix<float> Blob<Dtype>::getDataAsMatrix(unsigned int rows, unsigned int cols, bool isCollumnMajor) {
  phast::matrix<float> mat;
  return mat;
}

template <typename Dtype>
phast::matrix<float> Blob<Dtype>::getDiffAsMatrix(unsigned int rows, unsigned int cols, bool isCollumnMajor) {
  phast::matrix<float> mat;
  return mat;
}

template <typename Dtype>
phast::cube<float> Blob<Dtype>::getDataAsCube(unsigned int i, unsigned int j, unsigned int k) {
  phast::cube<float> cube;
  return cube;
}

template <typename Dtype>
phast::cube<float> Blob<Dtype>::getDiffAsCube(unsigned int i, unsigned int j, unsigned int k) {
  phast::cube<float> cube;
  return cube;
}


template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

template class Blob<float>;
template class Blob<double>;
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe



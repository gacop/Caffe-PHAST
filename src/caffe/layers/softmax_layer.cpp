#include <algorithm>
#include <vector>
#include <phast.h>

#include "caffe/phast_functors.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/configuration_file.h"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  softmax_axis_ = 1;
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

template <>
void SoftmaxLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  int channels = bottom[0]->shape(softmax_axis_);

  phast::cube<float> bottom_data = bottom[0]->getDataAsCube(outer_num_, channels, inner_num_);
  phast::cube<float> top_data = top[0]->getDataAsCube(outer_num_, channels, inner_num_);
  phast::matrix<float> scale_data = scale_.getDataAsMatrix(outer_num_, inner_num_, false); // la segunda dimension es la de outer_num_

#if DEBUGPRINT
  bottom[0]->print(false, std::string("BOTTOMSOFT") + layer_param().name());
  top[0]->print(false, std::string("TOPSOFT") + layer_param().name());
  scale_.print(false, std::string("SCALESOFT") + layer_param().name());
#endif

  top_data.assign(bottom_data);
  phast::configuration_file::retrieve_parameters("sm_transpose_ikj");
  top_data.transpose_ikj();

  phast::configuration_file::retrieve_parameters("sm_for_each3");
  phast::for_each(top_data.begin_ij(), top_data.end_ij(), scale_data.begin_ij(), func_softmax<float>());

  phast::configuration_file::retrieve_parameters("sm_transpose_ikj");
  top_data.transpose_ikj();

#if DEBUGPRINT
  bottom[0]->print(false, std::string("BOTTOMSOFT") + layer_param().name());
  top[0]->print(false, std::string("TOPSOFT") + layer_param().name());
  scale_.print(false, std::string("SCALESOFT") + layer_param().name());
#endif
}

template <>
void SoftmaxLayer<float>::Backward_cpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom) {

  int channels = top[0]->shape(softmax_axis_);
    
  phast::cube<float> topDiff = top[0]->getDiffAsCube(outer_num_, channels, inner_num_);
  phast::cube<float> topData = top[0]->getDataAsCube(outer_num_, channels, inner_num_);

  phast::cube<float> botDiff = bottom[0]->getDiffAsCube(outer_num_, channels, inner_num_);
  phast::vector<float> scaler = scale_.getDataAsVector(inner_num_);

  phast::copy(topDiff.begin_ijk(), topDiff.end_ijk(), botDiff.begin_ijk());

  auto botIt = botDiff.begin_i();
  auto topDIt = topData.begin_i();

  for (; botIt != botDiff.end_i() && topDIt != topData.end_i(); botIt++, topDIt++) {
    phast::matrix<float> botM;
    phast::matrix<float> topDM;
    botM.set_dev(botDiff.size_j(), botDiff.size_k(), botIt.get_dev() + botIt.get_abs_pos());
    topDM.set_dev(topData.size_j(), topData.size_k(), topDIt.get_dev() + topDIt.get_abs_pos());

    botM.transpose();
    topDM.transpose();
    
    reduceMatrixVectorByVectorDot<float> reduceMatrixVectorByVectorDot;
    reduceMatrixVectorByVectorDot.scal.link(scaler);
    phast::configuration_file::retrieve_parameters("sm_for_each4");
    phast::for_each(botM.begin_i(), botM.end_i(), topDM.begin_i(), reduceMatrixVectorByVectorDot);

    botM.transpose();
    topDM.transpose();

    matrixMinusVectorRows<float> matrixMinusVectorRows;
    matrixMinusVectorRows.vec.link(scaler);
    phast::configuration_file::retrieve_parameters("sm_for_each5");
    phast::for_each(botM.begin_i(), botM.end_i(), matrixMinusVectorRows);
  }

  phast::configuration_file::retrieve_parameters("sm_for_each6");
  phast::for_each(botDiff.begin_ijk(), botDiff.end_ijk(), topData.begin_ijk(), botDiff.begin_ijk(), phast::multiplies<float>());
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}


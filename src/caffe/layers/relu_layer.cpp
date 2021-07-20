#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

#include <phast.h>
#include "caffe/phast_functors.hpp"

namespace caffe {

template <>
void ReLULayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {

  const int count = bottom[0]->count();
  float negative_slope = this->layer_param_.relu_param().negative_slope();

  phast::vector<float> input = bottom[0]->getDataAsVector(count);
  phast::vector<float> output = top[0]->getDataAsVector(count);

  phast::for_each(input.begin(), input.end(), output.begin(), reluFunc<float>(negative_slope));
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope(); 
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <>
void ReLULayer<float>::Backward_cpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    float negative_slope = this->layer_param_.relu_param().negative_slope();

    phast::vector<float> input = bottom[0]->getDataAsVector(count);
    phast::vector<float> diff = top[0]->getDiffAsVector(count);
    phast::vector<float> output = bottom[0]->getDiffAsVector(count);

    phast::for_each(input.begin(), input.end(), diff.begin(), output.begin(), reluBackFunc<float>(negative_slope));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe

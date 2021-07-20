#include <functional>
#include <utility>
#include <cfloat>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <phast.h>
#include <print_utility.h>

#include "caffe/phast_functors.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <>
void AccuracyLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {

  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);

  phast::matrix<float> labels = bottom[1]->getDataAsMatrix(outer_num_, inner_num_, false);
  phast::cube<float> data = bottom[0]->getDataAsCube(outer_num_, num_labels, inner_num_);
  data.transpose_ikj();
  phast::vector<float> out = top[0]->getDataAsVector(1);
  phast::vector<float> out1;
  phast::vector<float> num;

  if (top.size() > 1) {
    out1 = top[1]->getDataAsVector(top[1]->count());
    num = nums_buffer_.getDataAsVector(top[1]->count());
    phast::fill(out1.begin(), out1.end(), float(0));
    phast::fill(num.begin(), num.end(), float(0));
  }

  int count = outer_num_ * inner_num_; // Fix (need to be the loop count)
  float accuracy = 0;

  auto dataIT = data.begin_ij();
  for (auto it = labels.begin_ij(); it != labels.end_ij() && dataIT != data.end_ij(); ++it, ++dataIT) {
    doAccuracy<float> doAccuracy((int)*it, num_labels, has_ignore_label_, ignore_label_, top.size(), top_k_);
    doAccuracy.out1.link(out1);
    doAccuracy.num.link(num);
    accuracy += phast::accumulate_for_each(dataIT, dataIT + 1, doAccuracy);
  }

  out[0] = (count == 0) ? 0 : (accuracy / count);
  if (top.size() > 1) {
    phast::replace(num.begin(), num.end(), float(0), float(FLT_MAX));
    phast::for_each(out1.begin(), out1.end(), num.begin(), out1.begin(), phast::divides<float>());
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int dim = bottom[0]->count() / outer_num_;  
  Dtype accuracy = 0;  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int num_labels = bottom[0]->shape(label_axis_);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      const Dtype prob_of_true_class = bottom_data[i * dim
                                                   + label_value * inner_num_
                                                   + j];
      int num_better_predictions = -1;  // true_class also counts as "better"
      // Top-k accuracy
      for (int k = 0; k < num_labels && num_better_predictions < top_k_; ++k) {
        num_better_predictions +=
          (bottom_data[i * dim + k * inner_num_ + j] >= prob_of_true_class);
      }
      // check if there are less than top_k_ predictions
      if (num_better_predictions < top_k_) {
        ++accuracy;
        if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
      }
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = (count == 0) ? 0 : (accuracy / count);
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

  

#ifdef CPU_ONLY
STUB_GPU(AccuracyLayer);
#endif

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe

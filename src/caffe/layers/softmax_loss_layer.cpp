#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <phast.h>
#include "caffe/phast_functors.hpp"

#include <print_utility.h>
#include "caffe/util/configuration_file.h"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <>
void SoftmaxWithLossLayer<float>::Forward_cpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  const int num_labels = prob_.count() / (outer_num_ * inner_num_);
  int dim = prob_.count() / outer_num_;
    
  phast::matrix<float> labels = bottom[1]->getDataAsMatrix(outer_num_, inner_num_, false);
  phast::cube<float> data = prob_.getDataAsCube(outer_num_, num_labels, inner_num_);
  //std::cout << bottom[0]->count() << std::endl;
  //std::cout << outer_num_ << " " << dim << " " << inner_num_ << std::endl;
  data.transpose_ikj();
  phast::vector<float> out = top[0]->getDataAsVector(1);

#if DEBUGPRINT
  bottom[1]->print(false, std::string("BOTTOM-") + this->layer_param_.name());
  prob_.print(false, std::string("PROB-") + this->layer_param_.name());
  top[0]->print(false, std::string("TOP-") + this->layer_param_.name());
#endif
  
  int count = outer_num_ * inner_num_; // Fix (Need iterations)

  phast::configuration_file::retrieve_parameters("sm_loss_acc_for_each1");
  float loss = phast::accumulate_for_each(data.begin_ij(), data.end_ij(), 
	doSMLoss<float>(labels, prob_.shape(softmax_axis_), has_ignore_label_, ignore_label_));

  out[0] = loss / get_normalizer(normalization_, count);
  data.transpose_ikj(); // ????
  if (top.size() == 2)
	top[1]->ShareData(prob_);

#if DEBUGPRINT
  bottom[1]->print(false, std::string("BOTTOM-") + this->layer_param_.name());
  prob_.print(false, std::string("PROB-") + this->layer_param_.name());
  top[0]->print(false, std::string("TOP-") + this->layer_param_.name());
#endif
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <>
void SoftmaxWithLossLayer<float>::Backward_cpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  if (propagate_down[1]) 
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  if (!propagate_down[0]) return;

  const int num_labels = prob_.count() / (outer_num_ * inner_num_);
    
  phast::matrix<float> labels = bottom[1]->getDataAsMatrix(outer_num_, inner_num_, false);
  phast::cube<float> prob_data = prob_.getDataAsCube(outer_num_, num_labels, inner_num_);
  phast::cube<float> bottomc = bottom[0]->getDiffAsCube(outer_num_, num_labels, inner_num_);
#if DEBUGPRINT
  bottom[1]->print(false, std::string("BACK-BOTTOM-") + this->layer_param_.name());
  prob_.print(false, std::string("BACK-PROB-") + this->layer_param_.name());
  bottom[0]->print(true, std::string("BACK-BOTTOM-DIFF-") + this->layer_param_.name());
  top[0]->print(true, std::string("BACK-TOP-DIFF-") + this->layer_param_.name());
#endif

  phast::copy(prob_data.begin_ijk(), prob_data.end_ijk(), bottomc.begin_ijk());
  bottomc.transpose_ikj();

  phast::configuration_file::retrieve_parameters("sm_loss_acc_for_each2");
  float count = phast::accumulate_for_each(bottomc.begin_ij(), bottomc.end_ij(), 
	doSMLossBack<float>(labels, prob_.shape(softmax_axis_), has_ignore_label_, ignore_label_));
  bottomc.transpose_ikj();
  
  // Scale gradient
  phast::vector<float> out = top[0]->getDiffAsVector(1);
  float scale = out[0] / get_normalizer(normalization_, count);
  phast::configuration_file::retrieve_parameters("sm_loss_for_each");
  phast::for_each(bottomc.begin_ijk(), bottomc.end_ijk(), vectorScale<float>(scale));
#if DEBUGPRINT
  bottom[0]->print(true, std::string("BACK-BOTTOM-DIFF-") + this->layer_param_.name());
#endif
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe

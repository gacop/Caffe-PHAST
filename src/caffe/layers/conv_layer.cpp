#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include <phast.h>
#include "caffe/phast_functors.hpp"
#include "phast_ai.h"
#include <print_utility.h>
#include "caffe/util/configuration_file.h"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <>
void ConvolutionLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  // Sanity checks
  if(this->stride_.count() > 1 && (this->stride_.cpu_data()[0] != this->stride_.cpu_data()[1])) {
    std::cerr << "FATAL: Convolution stride_w != stride_h" << std::endl;
    return;
  }
  if(this->dilation_.count() > 1 && (this->dilation_.cpu_data()[0] > 1 || this->dilation_.cpu_data()[1] > 1)) {
    std::cerr << "FATAL: Dilated Convolution is unimplemented!" << std::endl;
    return;
  }
  if (is_1x1_) {
    std::cerr << "FATAL: 1 x 1 Convolution is unimplemented!" << std::endl;
  }
  if (group_ > 1) {
    std::cerr << "FATAL: Grouped Convolution is unimplemented!" << std::endl;
    return;
  }
  if (this->force_nd_im2col_) {
    std::cerr << "FATAL: UNIMPLEMENTED 3 (" << this->force_nd_im2col_ << ")" << std::endl;
    return;
  }
  if(this->num_spatial_axes_ != 2) {
    std::cerr << "UNIMPLEMENTED 4" << std::endl;
    return;
  }
#if DEBUGPRINT
    if(this->bias_term_)
        this->blobs_[1]->print(false, std::string("BIAS-") + layer_param().name());
    this->blobs_[0]->print(false, std::string("FILTERS-") + layer_param().name());
#endif

  // Prepare data
  vector<int> filters_shape = this->blobs_[0]->shape();
  int stride = this->stride_.cpu_data()[0];
  int filter_x = filters_shape[2];
  int filter_y = filters_shape[3];
  phast::vector<float> bias, *bias_ptr = nullptr;
  if (this->bias_term_)
  {
	bias = this->blobs_[1]->getDataAsVector(this->conv_out_channels_);
    bias_ptr = &bias;
  }

  phast::cube<float> filters = this->blobs_[0]->getDataAsCube(this->conv_out_channels_ * this->conv_in_channels_, filter_x, filter_y);

  for (int i = 0; i < bottom.size(); ++i) {
#if DEBUGPRINT
    bottom[i]->print(false, std::string("BOTTOM-") + layer_param().name());
    top[i]->print(false, std::string("TOP-") + layer_param().name());
#endif
    vector<int> top_shape = top[i]->shape();
    int top_x = top_shape[2];
    int top_y = top_shape[3];

    phast::cube<float> bottom_data = bottom[i]->getDataAsCube(this->num_ * this->conv_in_channels_, this->conv_input_shape_.cpu_data()[1], this->conv_input_shape_.cpu_data()[2]);
    phast::cube<float> top_data = top[i]->getDataAsCube(this->num_ * this->conv_out_channels_, top_x, top_y);

	phast::configuration_file::retrieve_parameters(std::string("forward_conv_") + layer_param().name());
	phast::ai::batch_convolution(bottom_data, filters, this->conv_out_channels_, stride, bias_ptr, top_data);
#if DEBUGPRINT
    top[i]->print(false, std::string("TOP-") + layer_param().name());
#endif

  }

}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}


template <>
void ConvolutionLayer<float>::Backward_cpu(const vector<Blob<float>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  if(this->stride_.count() > 1 && (this->stride_.cpu_data()[0] != this->stride_.cpu_data()[1])) {
    std::cerr << "FATAL: Convolution stride_w != stride_h" << std::endl;
    return;
  }
  if (group_ > 1) {
    std::cerr << "FATAL: Grouped Convolution is unimplemented!" << std::endl;
    return;
  }
  if (is_1x1_) {
    std::cerr << "FATAL: 1 x 1 Convolution is unimplemented!" << std::endl;
    return;
  }

  int stride = this->stride_.cpu_data()[0];
  phast::vector<float> *bias_dummy = nullptr;

  for (int i = 0; i < top.size(); i++) {
    // Step 0. Prepare data
    vector<int> d_shp = this->blobs_[0]->shape();
    vector<int> b_shp = bottom[i]->shape();
    vector<int> t_shp = top[i]->shape();

    // Step 1. Bias
    if (this->bias_term_ && this->param_propagate_down_[1]) {
#if DEBUGPRINT
	  top[i]->print(true, std::string("BACK-TOP-") + layer_param().name());
	  this->blobs_[1]->print(true, std::string("BACK-BIAS-") + layer_param().name());
#endif

      phast::cube<float> top_diff_tmp = top[i]->getDiffAsCube(this->num_ * this->conv_out_channels_, t_shp[2], t_shp[3]);
      phast::grid<phast::cube<float>> top_diff(top_diff_tmp, 1, t_shp[2], t_shp[3]);
      phast::vector<float> bias_diff = this->blobs_[1]->getDiffAsVector(this->num_output_);
	  phast::matrix<float> tmp_acc(this->num_, this->conv_out_channels_);

	  phast::configuration_file::retrieve_parameters(std::string("backward_bias1_") + layer_param().name());
	  phast::for_each(top_diff.begin(), top_diff.end(), tmp_acc.begin_ij(), func_conv_bp_bias<float>());
	  tmp_acc.transpose();
	  phast::configuration_file::retrieve_parameters(std::string("backward_bias2_") + layer_param().name());
	  phast::for_each(tmp_acc.begin_i(), tmp_acc.end_i(), bias_diff.begin(), reduceMatrixVectors<float>());
#if DEBUGPRINT
	  this->blobs_[1]->print(true, std::string("BACK-BIAS-") + layer_param().name());
#endif
    }

	if(this->param_propagate_down_[0] || propagate_down[i])
	{
		phast::cube<float> top_diff = top[i]->getDiffAsCube(t_shp[0]*t_shp[1], t_shp[2], t_shp[3]);

	    // Step 2. Weights
	    if(this->param_propagate_down_[0]) {
#if DEBUGPRINT
		  this->blobs_[0]->print(true, std::string("BACK-WEIGHT-DIFF-") + layer_param().name());
		  bottom[i]->print(false, std::string("BACK-IMAGES-") + layer_param().name());
#endif
	      phast::cube<float> weight_diff = this->blobs_[0]->getDiffAsCube(d_shp[0] * d_shp[1], d_shp[2], d_shp[3]);
		  phast::cube<float> images = bottom[i]->getDataAsCube(b_shp[0]*b_shp[1], b_shp[2], b_shp[3]);
	
		  phast::configuration_file::retrieve_parameters(std::string("backward_conv_") + layer_param().name());
	      phast::ai::batch_convolution_channel_major(images, top_diff, t_shp[1], stride, bias_dummy, weight_diff);
#if DEBUGPRINT
		  this->blobs_[0]->print(true, std::string("BACK-WEIGHT-DIFF-OUT-") + layer_param().name());
#endif
	    }
	
	    // Step 3. Bottom Data
	    if (propagate_down[i]) {
#if DEBUGPRINT
			bottom[i]->print(true, std::string("BACK-BOTTOM-DIFF-") + layer_param().name());
			this->blobs_[0]->print(false, std::string("BACK-WEIGHT-") + layer_param().name());
#endif
		    phast::cube<float> weights = this->blobs_[0]->getDataAsCube(d_shp[0] * d_shp[1], d_shp[2], d_shp[3]);
		    phast::cube<float> bottom_diff = bottom[i]->getDiffAsCube(b_shp[0] * b_shp[1], b_shp[2], b_shp[3]);

			phast::configuration_file::retrieve_parameters(std::string("backward_transp_") + layer_param().name());
			phast::cube<float> weights_tr(weights.size_i(), weights.size_j(), weights.size_k());
			phast::for_each(weights_tr.begin_i(), weights_tr.end_i(), transposer<float>(weights, d_shp[0], d_shp[1]));

			phast::configuration_file::retrieve_parameters(std::string("backward_wpad_") + layer_param().name());
			phast::ai::rotate_and_pad(weights_tr, 0, 0, rotated_weights_);
			phast::configuration_file::retrieve_parameters(std::string("backward_tpad_") + layer_param().name());
			const int pad_h = (padded_top_.size_j() - top_diff.size_j()) >> 1;
			const int pad_w = (padded_top_.size_k() - top_diff.size_k()) >> 1;
			phast::ai::pad(top_diff, pad_h, pad_w, padded_top_);

			phast::configuration_file::retrieve_parameters(std::string("backward_conv_bottom_") + layer_param().name());
			phast::ai::batch_convolution(padded_top_, rotated_weights_, d_shp[1], stride, bias_dummy, bottom_diff);

#if DEBUGPRINT
			bottom[i]->print(true, std::string("BACK-BOTTOM-DIFF-OUT-") + layer_param().name());
#endif
	    }
	}
  }

}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
  
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <phast.h>
#include <print_utility.h>

#include "caffe/phast_functors.hpp"
#include "caffe/util/configuration_file.h"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <>
void InnerProductLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {

  // Data as Matrix
  phast::matrix<float> matA = bottom[0]->getDataAsMatrix(M_, K_, false);
  phast::matrix<float> matB = this->blobs_[0]->getDataAsMatrix(K_, N_, !transpose_);
  phast::matrix<float> matC = top[0]->getDataAsMatrix(M_, N_, false);

#if DEBUGPRINT
	bottom[0]->print(false, std::string("BOTTOM-") + this->layer_param_.name());
	this->blobs_[0]->print(false, std::string("BLOB-") + this->layer_param_.name());
	top[0]->print(false, std::string("TOP-") + this->layer_param_.name());
#endif

  // Matrix Mul Inputs x Weights = Output  
  phast::configuration_file::retrieve_parameters(std::string("dot_prod1_") + layer_param().name());
  phast::dot_product(matA, matB, matC);

  // Add bias by to Output by Rows
  if (bias_term_) {
    matrixPlusVectorRows<float> matrixPlusVectorRows;
    matrixPlusVectorRows.vec.link(this->blobs_[1]->getDataAsVector(N_));
	phast::configuration_file::retrieve_parameters(std::string("for_each1_") + layer_param().name());
    phast::for_each(matC.begin_i(), matC.end_i(), matrixPlusVectorRows);
  }
  if(!transpose_) matB.transpose(); // A way to tranpose matrix without transposing data ?

#if DEBUGPRINT
	this->blobs_[0]->print(false, std::string("BLOB-OUT-") + this->layer_param_.name());
	top[0]->print(false, std::string("TOP-OUT-") + this->layer_param_.name());
#endif
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <>
void InnerProductLayer<float>::Backward_cpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom) {

  if (this->param_propagate_down_[0]) {
#if DEBUGPRINT
	top[0]->print(true, std::string("BACK-TOPDIFF-") + this->layer_param_.name());
	bottom[0]->print(false, std::string("BACK-BOTTOM-") + this->layer_param_.name());
	this->blobs_[0]->print(true, std::string("BACK-WDIFF-") + this->layer_param_.name());
#endif
    if (transpose_) {
      phast::matrix<float> diff = top[0]->getDiffAsMatrix(M_, N_, false);
      phast::matrix<float> data = bottom[0]->getDataAsMatrix(K_, M_, true);
      phast::matrix<float> tmp(K_, N_, 0);
      phast::matrix<float> wdiff = this->blobs_[0]->getDiffAsMatrix(K_, N_, false);

  	  phast::configuration_file::retrieve_parameters(std::string("dot_prod2_") + layer_param().name());
      phast::dot_product(data, diff, tmp);
	  phast::configuration_file::retrieve_parameters(std::string("for_each2_") + layer_param().name());
      phast::for_each(tmp.begin_ij(), tmp.end_ij(), wdiff.begin_ij(), wdiff.begin_ij(),
                       phast::plus<float>());
      data.transpose();
    }
    else {
      //float* tmp_ptr1 = (float *) this->blobs_[0]->cpu_diff();
      phast::matrix<float> diff = top[0]->getDiffAsMatrix(N_, M_, true);
      phast::matrix<float> data = bottom[0]->getDataAsMatrix(M_, K_, false);
      phast::matrix<float> tmp(N_, K_, 0);
      phast::matrix<float> wdiff = this->blobs_[0]->getDiffAsMatrix(N_, K_, false);
      //float* tmp_ptr2 = (float *) this->blobs_[0]->cpu_diff();
      //assert(tmp_ptr1 == tmp_ptr2);
      //
  	  phast::configuration_file::retrieve_parameters(std::string("dot_prod3_") + layer_param().name());
      phast::dot_product(diff, data, tmp);
	  phast::configuration_file::retrieve_parameters(std::string("for_each3_") + layer_param().name());
      phast::for_each(tmp.begin_ij(), tmp.end_ij(), wdiff.begin_ij(), wdiff.begin_ij(),
                       phast::plus<float>());
      diff.transpose();
    }
#if DEBUGPRINT
	this->blobs_[0]->print(true, std::string("BACK-WDIFF-OUT-") + this->layer_param_.name());
#endif
  }
  this->blobs_[0]->syncDiffToLocal();
  if (bias_term_ && this->param_propagate_down_[1]) {
#if DEBUGPRINT
	top[0]->print(true, std::string("BACK-TOPDIFF-") + this->layer_param_.name());
	this->blobs_[1]->print(true, std::string("BACK-WDIFF-1-") + this->layer_param_.name());
#endif
    phast::matrix<float> diff = top[0]->getDiffAsMatrix(N_, M_, true);
    phast::vector<float> weig = this->blobs_[1]->getDiffAsVector(N_);

	phast::configuration_file::retrieve_parameters(std::string("for_each4_") + layer_param().name());
    phast::for_each(diff.begin_i(), diff.end_i(), weig.begin(), reduceMatrixVectors<float>());
    
    diff.transpose();
#if DEBUGPRINT
	top[0]->print(true, std::string("BACK-TOPDIFF-OUT-") + this->layer_param_.name());
#endif
  }
  if (propagate_down[0]) {
#if DEBUGPRINT
	bottom[0]->print(true, std::string("BACK-BOTTOMDIFF-") + this->layer_param_.name());
	this->blobs_[0]->print(false, std::string("BACK-WEIGH-") + this->layer_param_.name());
#endif
    phast::matrix<float> diff = top[0]->getDiffAsMatrix(M_, N_, false);
    phast::matrix<float> weig = this->blobs_[0]->getDataAsMatrix(N_, K_, false);
    phast::matrix<float> bdiff = bottom[0]->getDiffAsMatrix(M_, K_, false);

	auto weig_(weig);
	if(transpose_)
		weig_.transpose();

  	phast::configuration_file::retrieve_parameters(std::string("dot_prod4_") + layer_param().name());
    phast::dot_product(diff, weig_, bdiff);
#if DEBUGPRINT
	bottom[0]->print(true, std::string("BACK-BOTTOMDIFF-OUT-") + this->layer_param_.name());
#endif
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasTrans : CblasNoTrans,
                          M_, K_, N_,
                          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                          (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe

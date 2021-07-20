#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/phast_functors.hpp"
#include "caffe/util/configuration_file.h"

struct mega_elapsed
{
	mega_elapsed() : elapsed(0.0) {}

	void start()
	{
		t.start();
	}
	void stop()
	{
		t.stop();
		elapsed += t.get_elapsed();
	}
	~mega_elapsed()
	{
		std::cout << "ADAM] Total Time: " << elapsed << " ms" << std::endl;
	}

	phast::timer t;
	double elapsed;
};

static mega_elapsed mega_elapsed_;

namespace caffe {

template <typename Dtype>
void AdamSolver<Dtype>::AdamPreSolve() {
  // Add the extra history entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void adam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1,
    Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate);
#endif

template <typename T, unsigned int policy = phast::get_default_policy()>
struct solver_axbpy : phast::functor::func_scal_scal<T, policy> {

  T a;
  T b;

  _PHAST_METHOD solver_axbpy(T a, T b) {
    this->a = a;
    this->b = b;
  }

  _PHAST_METHOD void operator()(const phast::functor::scalar<T>& X, phast::functor::scalar<T>& Y) {
    Y = a * X + b * Y;
  }

};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct solver_axbpy_square : phast::functor::func_scal_scal<T, policy> {

  T * square_data;
  T a;
  T b;

  _PHAST_METHOD solver_axbpy_square(T a, T b, T * net_params) {
    this->a = a;
    this->b = b;
    this->square_data = net_params;
  }

  _PHAST_METHOD void operator()(phast::functor::scalar<T>& X, phast::functor::scalar<T>& Y) {
    int i = this->get_index();
    X = square_data[i] * square_data[i];
    Y = a * X + b * Y;
  }

};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct solver_update : phast::functor::func_scal_scal<T, policy> {

  T pow;
  T a;
  T * val_m;

  _PHAST_METHOD solver_update(T pow, T a, T * val_m) {
    this->pow = pow;
    this->a = a;
    this->val_m = val_m;
  }

  _PHAST_METHOD void operator()(phast::functor::scalar<T>& X, phast::functor::scalar<T>& Y) {
    int i = this->get_index();
    Y = val_m[i] / (phast::math::pow(X, this->pow) + this->a);
  }

};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct solver_scale : phast::functor::func_scal_scal<T, policy> {

  T val;

  _PHAST_METHOD solver_scale(T val) {
    this->val = val;
  }

  _PHAST_METHOD void operator()(const phast::functor::scalar<T>& X,phast::functor::scalar<T>& Y) {
    Y = X * this->val;
  }

};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct adam_solver : phast::functor::func_scal<T, policy>
{
	_PHAST_METHOD adam_solver(phast::vector<T>& val_m, phast::vector<T>& val_v, phast::vector<T>& val_t,
		T beta1, T beta2, T a, T scale) : beta1_(beta1), beta2_(beta2), a_(a), scale_(scale)
	{
		val_m_.link(val_m);
		val_v_.link(val_v);
		val_t_.link(val_t);
	}

	_PHAST_METHOD void operator()(phast::functor::scalar<T>& np_diff)
	{
		int i = this->get_index();

		val_m_[i] = (1 - beta1_)*np_diff + beta1_*val_m_[i];
		val_v_[i] = (1 - beta2_)*np_diff*np_diff + beta2_*val_v_[i];
    	val_t_[i] = val_m_[i] / (phast::math::sqrt(val_v_[i]) + a_);
		np_diff = val_t_[i] * scale_;
	}

	phast::functor::vector<T> val_m_;
	phast::functor::vector<T> val_v_;
	phast::functor::vector<T> val_t_;
	T beta1_, beta2_, a_, scale_;
};

void debug_fill_vector(float* v, int n) {
  for(int i=0; i < n; i++)
    v[i] = ((float) i)/10;
}

void debug_empty_vector(float* v, int n) {
  for(int i=0; i < n; i++)
    v[i] = 0.0;
}

// this->net_->learnable_params()
// net_params[param_id]->mutable_cpu_diff()
// TODO: Instead of this->get_index(), use func_scal_scal_scal
template <>
void AdamSolver<float>::ComputeUpdateValue(int param_id, float rate) {

  mega_elapsed_.start();

  const vector<Blob<float>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  float local_rate = rate * net_params_lr[param_id];
  float beta1 = this->param_.momentum();
  float beta2 = this->param_.momentum2();
  int update_history_offset = net_params.size();
  const int N = net_params[param_id]->count();

  Blob<float>* val_m_ptr = this->history_[param_id].get();
  Blob<float>* val_v_ptr = this->history_[param_id + update_history_offset].get();
  Blob<float>* val_t_ptr = this->temp_[param_id].get();

  /* debug_fill_vector((float *) net_params[param_id]->cpu_diff(), N);
  debug_empty_vector(val_m_ptr->mutable_cpu_data(), N);
  debug_empty_vector(val_v_ptr->mutable_cpu_data(), N);
  debug_empty_vector(val_t_ptr->mutable_cpu_data(), N); */

  /*phast::vector<float> val_m;
  phast::vector<float> val_v;
  phast::vector<float> val_t;
  phast::vector<float> np_diff;
  val_m.set_dev(N, val_m_ptr->mutable_cpu_data());
  val_v.set_dev(N, val_v_ptr->mutable_cpu_data());
  val_t.set_dev(N, val_t_ptr->mutable_cpu_data());
  np_diff.set_dev(N, (float *) net_params[param_id]->cpu_diff());*/
  phast::vector<float> val_m = val_m_ptr->getDataAsVector(N);
  phast::vector<float> val_v = val_v_ptr->getDataAsVector(N);
  phast::vector<float> val_t = val_t_ptr->getDataAsVector(N);
  phast::vector<float> np_diff = net_params[param_id]->getDiffAsVector(N);

  const int t = this->iter_ + 1;
  const float correction = std::sqrt(1 - pow(beta2, t)) / (1.0 - pow(beta1, t));
  const float eps_hat = this->param_.delta();

  phast::configuration_file::retrieve_parameters("adam");
  phast::for_each(np_diff.begin(), np_diff.end(), 
	adam_solver<float>(val_m, val_v, val_t, beta1, beta2, eps_hat, local_rate*correction));

  mega_elapsed_.stop();
}

template <typename Dtype>
void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum();
  const Dtype beta2 = this->param_.momentum2();

  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_t = this->temp_[param_id].get();

  const int t = this->iter_ + 1;
  const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
      (Dtype(1.) - pow(beta1, t));
  const int N = net_params[param_id]->count();
  const Dtype eps_hat = this->param_.delta();

  switch (Caffe::mode()) {
    case Caffe::CPU: {
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_cpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->cpu_diff(), beta1,
        val_m->mutable_cpu_data());

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_mul(N,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->cpu_diff(),
    val_t->mutable_cpu_data());
    caffe_cpu_axpby(N, Dtype(1)-beta2,
        val_t->cpu_data(), beta2,
        val_v->mutable_cpu_data());

    // set update (un solo functor porque se escribe en val_t)
    caffe_powx(N,
        val_v->cpu_data(), Dtype(0.5),
        val_t->mutable_cpu_data());
    caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
    caffe_div(N,
        val_m->cpu_data(),
        val_t->cpu_data(),
        val_t->mutable_cpu_data());

    // esto va por separado
    caffe_cpu_scale(N, local_rate*correction,
        val_t->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    adam_update_gpu(N, net_params[param_id]->mutable_gpu_diff(),
        val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), beta1, beta2,
        eps_hat, local_rate*correction);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(AdamSolver);
REGISTER_SOLVER_CLASS(Adam);

}  // namespace caffe

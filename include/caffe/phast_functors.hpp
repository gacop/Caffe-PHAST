#ifndef __PHAST_FUNCTORS_HPP__
#define __PHAST_FUNCTORS_HPP__

#include <phast.h>
#include <cfloat>

#define smax(x, y) (((x) >= (y))? (x) : (y))
#define smin(x, y) (((x) <= (y))? (x) : (y))

template <typename T, unsigned int policy = phast::get_default_policy()>
struct func_exp : phast::functor::func_scal_scal<T, policy>
{
    _PHAST_METHOD func_exp() { }

    _PHAST_METHOD void operator()(phast::functor::scalar<T>& top_data, phast::functor::scalar<T>& bottom_data) {
      top_data = phast::math::exp(bottom_data);
    }
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct func_scale : phast::functor::func_mat_vec<T, policy>
{
    int inner_num_;
    int channels;

    _PHAST_METHOD func_scale(int i, int j) {
      inner_num_ = i;
      channels = j;
    }

    // otro functor aqui (doble funtor) mas eficiente en gpu
    _PHAST_METHOD void operator()(phast::functor::matrix<T>& top_data_face, phast::functor::vector<T>& scale_data) {
      for(int i = 0; i < inner_num_; i++) {
        scale_data[i] = 0;
        for(int j=0; j < channels; j++) {
          scale_data[i] += top_data_face.at(j,i); // No puedo usar accumulate porque necesito recorrer por columnas
        }
      }
    }
};

template <typename T>
struct inner_softmax : phast::inner_functor::func_scal<T>
{
	_PHAST_METHOD inner_softmax(T val) : val_(val) {}
	_PHAST_METHOD void operator()(phast::inner_functor::scalar<T>& scal)
	{
		scal = phast::math::exp(scal - val_);
	}

	T val_;
};

template <typename T>
struct inner_div : phast::inner_functor::func_scal<T>
{
	_PHAST_METHOD inner_div(T val) : val_(val) {}
	_PHAST_METHOD void operator()(phast::inner_functor::scalar<T>& scal)
	{
		scal /= val_;
	}

	T val_;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct func_softmax : phast::functor::func_vec_scal<T, policy>
{
    _PHAST_METHOD void operator()(phast::functor::vector<T>& row, phast::functor::scalar<T>& scal)
    {
        scal = smax(scal, *this->max_element(row.begin(), row.end()));
		this->for_each(row.begin(), row.end(), inner_softmax<T>(scal));
		scal = this->accumulate(row.begin(), row.end());
		this->for_each(row.begin(), row.end(), inner_div<T>(scal));
    }
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct func_div : phast::functor::func_mat<T, policy>
{
    phast::functor::matrix<T> scale_data;
    int inner_num_;
    int channels;

    _PHAST_METHOD func_div(phast::matrix<T>& scale_data, int inner_num__, int channels_) { 
      this->scale_data.link(scale_data); 
      channels = channels_;
      inner_num_ = inner_num__;
    }

    _PHAST_METHOD void operator()(phast::functor::matrix<T>& top_data) {
      for(int i = 0; i < channels; i++) {
        for(int j = 0; j < inner_num_; j++) {
           top_data.at(i,j) = top_data.at(i,j) / scale_data[this->get_index()][j];
        }
      }
    }

};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct vectorScale : phast::functor::func_scal<T, policy> {

  _PHAST_METHOD vectorScale(T alpha)
    : alpha_(alpha) {}

  _PHAST_METHOD void operator()(phast::functor::scalar<T>& scal) {
    scal *= alpha_;
  }

  T alpha_;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct matrixPlusVectorRows : phast::functor::func_vec<T, policy> {

  _PHAST_METHOD matrixPlusVectorRows() {}
  
  _PHAST_METHOD void operator()(phast::functor::vector<T>& row) {
    for(auto r = row.begin(), i = vec.begin(); r != row.end(); ++r, ++i)
      *r += *i;
  }
  
  phast::functor::vector<T> vec;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct matrixMinusVectorRowsExp : phast::functor::func_vec<T, policy> {

  _PHAST_METHOD matrixMinusVectorRowsExp() {}
  
  _PHAST_METHOD void operator()(phast::functor::vector<T>& row) {
    auto r = row.begin();
    auto i = vec.begin();
    for(; r != row.end() && i != vec.end(); ++r, ++i)
      *r = phast::math::exp(*r - *i);
  }
  
  phast::functor::vector<T> vec;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct matrixMinusVectorRows : phast::functor::func_vec<T, policy> {

  _PHAST_METHOD matrixMinusVectorRows() {}
  
  _PHAST_METHOD void operator()(phast::functor::vector<T>& row) {
    auto r = row.begin();
    auto i = vec.begin();
    for(; r != row.end() && i != vec.end(); ++r, ++i)
      *r -= *i;
  }
  
  phast::functor::vector<T> vec;
};

template <typename T>
struct matrixPlusVectorColsInnerFunc : phast::inner_functor::func_scal<T>
{
  _PHAST_METHOD void operator()(phast::inner_functor::scalar<T>& row_elem) {
	row_elem += scal;
  }

  T scal;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct matrixPlusVectorCols : phast::functor::func_scal_vec<T, policy> {

  _PHAST_METHOD matrixPlusVectorCols() {}
  
  _PHAST_METHOD void operator()(const phast::functor::scalar<T>& scal, phast::functor::vector<T>& row) 
  {
    innerFunctor.scal = scal;
    this->for_each(row.begin(), row.end(), innerFunctor);
  }

  matrixPlusVectorColsInnerFunc<T> innerFunctor;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct reduceMatrixVectors : phast::functor::func_vec_scal<T, policy> {

  _PHAST_METHOD reduceMatrixVectors() {}

  _PHAST_METHOD void operator()(const phast::functor::vector<T>& row, phast::functor::scalar<T>& scal) {
    scal += this->accumulate(row.begin(), row.end());
  }
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct matrixDivVectorRows : phast::functor::func_vec<T, policy> {

  _PHAST_METHOD matrixDivVectorRows() {}
  
  _PHAST_METHOD void operator()(phast::functor::vector<T>& row) {
    auto r = row.begin();
    auto d = vec.begin();
    for(; r != row.end() && d != vec.end(); ++r, ++d)
      *r = (*r)/(*d);
  }

  phast::functor::vector<T> vec;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct matrixReduceByRows : phast::functor::func_scal_vec<T, policy> {

  _PHAST_METHOD matrixReduceByRows() {}
  
  _PHAST_METHOD void operator()(phast::functor::scalar<T>& scal,phast::functor::vector<T>& row) {
    scal = this->accumulate(row.begin(), row.end());
  }
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct reduceMatrixVectorByVectorDot : phast::functor::func_vec_vec<T, policy> {

  _PHAST_METHOD reduceMatrixVectorByVectorDot() {}

  _PHAST_METHOD void operator()(phast::functor::vector<T>& i, phast::functor::vector<T>& j) {
    int index = this->get_index();
    scal[index] = this->dot_product(i, j);
  }

  phast::functor::vector<T> scal;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct ppPoolingMax : phast::functor::func_scal_scal<T, policy>
{
    _PHAST_METHOD ppPoolingMax(phast::cube<T>& input_, const int n_tot_, const int n_row_, const int kh_, const int kw_, const int sh_,
        const int sw_, const int ph_, const int pw_) : n_tot(n_tot_), n_row(n_row_), kh(kh_), kw(kw_), sh(sh_), sw(sw_), ph(ph_), pw(pw_)
    {
        input.link(input_);
    }
    _PHAST_METHOD void operator()(phast::functor::scalar<T>& mask, phast::functor::scalar<T>& out)
    {
        const int index = this->get_index();
        const int I = index / n_tot;
        const int t = index % n_tot;
        const int J = -ph + (t / n_row)*sh;
        const int K = -pw + (t % n_row)*sw;

        T num = -FLT_MAX;
		mask = -1.0f;
        for(int y=0; y<kh; ++y)
        {
            for(int x=0; x<kw; ++x)
            {
                const T val = input.at(I, J + y, K + x);
                if(num < val)
                {
                    num = val;
                    mask = (J + y)*input.size_k() + (K + x);
                }
            }
        }
        out = num;
    }

    phast::functor::cube<T> input;
    const int n_tot, n_row, kh, kw, sh, sw, ph, pw;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct ppPoolingMaxBack : phast::functor::func_scal_scal<T, policy>
{
    _PHAST_METHOD ppPoolingMaxBack(phast::cube<T>& bottom_, const int n_tot_) : n_tot(n_tot_)
    {
        bottom.link(bottom_);
    }
    _PHAST_METHOD void operator()(const phast::functor::scalar<T>& mask, const phast::functor::scalar<T>& top)
    {
        const int I = this->get_index() / n_tot;

		const int index = mask;
		const int J = index / bottom.size_j();
		const int K = index % bottom.size_j();
		bottom.at(I, J, K) += top;
    }

    phast::functor::cube<T> bottom;
    const int n_tot;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct ppPoolingAvg : phast::functor::func_scal<T, policy>
{
    _PHAST_METHOD ppPoolingAvg(phast::cube<T>& input_, const int n_tot_, const int n_row_, const int kh_, const int kw_, const int sh_,
        const int sw_, const int ph_, const int pw_) : n_tot(n_tot_), n_row(n_row_), kh(kh_), kw(kw_), sh(sh_), sw(sw_), ph(ph_), pw(pw_)
    {
        input.link(input_);
    }
    _PHAST_METHOD void operator()(phast::functor::scalar<T>& out)
    {
        const int index = this->get_index();
        const int I = index / n_tot;
        const int t = index % n_tot;
        const int J = -ph + (t / n_row)*sh;
        const int K = -pw + (t % n_row)*sw;

        T num = 0.0f;
        for(int y=0; y<kh; ++y)
        {
            for(int x=0; x<kw; ++x)
            {
                num += input.at(I, J + y, K + x);
            }
        }
        out = num / (kh * kw);
    }

    phast::functor::cube<T> input;
    const int n_tot, n_row, kh, kw, sh, sw, ph, pw;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct ppPoolingAvgBack : phast::functor::func_scal<T, policy>
{
    _PHAST_METHOD ppPoolingAvgBack(phast::cube<T>& bottom_, const int n_tot_, const int n_row_, const int kh_, const int kw_, const int sh_,
        const int sw_, const int ph_, const int pw_) : n_tot(n_tot_), n_row(n_row_), kh(kh_), kw(kw_), sh(sh_), sw(sw_), ph(ph_), pw(pw_)
    {
        bottom.link(bottom_);
    }
    _PHAST_METHOD void operator()(phast::functor::scalar<T>& top)
    {
        const int index = this->get_index();
        const int I = index / n_tot;
        const int t = index % n_tot;
        const int J = -ph + (t / n_row)*sh;
        const int K = -pw + (t % n_row)*sw;

		const int KS = kh*kw;
        for(int y=0; y<kh; ++y)
        {
            for(int x=0; x<kw; ++x)
            {
                bottom.at(I, J + y, K + x) += top / KS;
            }
        }
    }

    phast::functor::cube<T> bottom;
    const int n_tot, n_row, kh, kw, sh, sw, ph, pw;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct im2Col : phast::functor::func_scal<T, policy> {

  _PHAST_METHOD im2Col(int dilation_h, int dilation_w,
                       int stride_h, int stride_w,
                       int pad_h, int pad_w,
                       int kernel_h, int kernel_w,
                       int height, int width) {
    dh = dilation_h; dw = dilation_w;
    sh = stride_h;   sw = stride_w;
    ph = pad_h;      pw = pad_w;
    kh = kernel_h;   kw = kernel_w;
    h = height;      w = width;

    oih = (h + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    oiw = (w + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
  }
  
  _PHAST_METHOD void operator()(phast::functor::scalar<T>& col) {
    int index = this->get_index();

    const int irow = -ph + (((index / (oih*oiw)) % (kh * kw)) / kw) * dh + ((index % (oih*oiw)) / oiw) * sh;
    const int icol = -pw + (((index / (oih*oiw)) % (kh * kw)) % kw) * dw + ((index % (oih*oiw)) % oiw) * sw;
    if (irow >= 0 && irow < h && icol >= 0 && icol < w)
      col = in.at((index / (oih*oiw*kh*kw)), irow * w + icol);
    else col = 0;
  }

  int dh, dw;
  int sh, sw;
  int ph, pw;
  int kh, kw;
  int  h,  w;
  int oih,oiw;

  phast::functor::matrix<T> in;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct col2Im : phast::functor::func_scal<T, policy> {

  _PHAST_METHOD col2Im(int dilation_h, int dilation_w,
                       int stride_h, int stride_w,
                       int pad_h, int pad_w,
                       int kernel_h, int kernel_w,
                       int height, int width) {
    dh = dilation_h; dw = dilation_w;
    sh = stride_h;   sw = stride_w;
    ph = pad_h;      pw = pad_w;
    kh = kernel_h;   kw = kernel_w;
    h = height;      w = width;

    oih = (h + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    oiw = (w + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
  }

  _PHAST_METHOD void operator()(phast::functor::scalar<T>& col) {
    int index = this->get_index();
    
    const int irow = -ph + (((index / (oih*oiw)) % (kh * kw)) / kw) * dh + ((index % (oih*oiw)) / oiw) * sh;
    const int icol = -pw + (((index / (oih*oiw)) % (kh * kw)) % kw) * dw + ((index % (oih*oiw)) % oiw) * sw;
    if (irow >= 0 && irow < h && icol >= 0 && icol < w)
      in.at((index / (oih*oiw*kh*kw)), irow * w + icol) = col;
  }

  int dh, dw;
  int sh, sw;
  int ph, pw;
  int kh, kw;
  int  h,  w;
  int oih,oiw;

  phast::functor::matrix<T> in;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct doAccuracy : phast::functor::func_vec<T, policy> {

  _PHAST_METHOD doAccuracy(int label_, int maxLabels, int ignore, int ignoreValue, int isTop, int top_k) {
    label = label_;
    labels = maxLabels;
    hasIgnore = ignore;
    value = ignoreValue;
    top = isTop;
    topk = top_k;
  }
  
  _PHAST_METHOD T operator()(phast::functor::vector<T>& row) {
    T acc = 0;
    if (hasIgnore && label == value) return 0;
    if (label < 0 || label > labels) return 0;

    if (top > 1) num[label]++;
    T prob = row[label];

    int predicts = -1;
    
    for (auto it = row.begin(); it != row.end() && predicts < topk; ++it) {
      if ((*it) >= prob) predicts++;
    }

    if (predicts < topk) {
      ++acc;
      if (top > 1) out1[label]++;
    }
    
    return acc;
  }

  phast::functor::vector<T> out1;
  phast::functor::vector<T> num;
  
  int label;
  int labels;
  int hasIgnore;
  int value;
  int top;
  int topk;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct doSMLoss : phast::functor::func_vec<T, policy> {

  _PHAST_METHOD doSMLoss(const phast::matrix<T>& labelMat_, int maxLabels, int ignore, int ignoreValue) {
    labelMat.link(labelMat_);
    labels = maxLabels;
    hasIgnore = ignore;
    value = ignoreValue;
  }
  
  _PHAST_METHOD T operator()(phast::functor::vector<T>& row) {
	const int label = (int)(*(labelMat.begin_ij() + this->get_index()));
    if (hasIgnore && label == value) return 0;
    if (label < 0 || label > labels) return 0;
    return -log(smax(row[label], T(FLT_MIN)));
  }
  
  phast::functor::matrix<T> labelMat;
  int labels;
  int hasIgnore;
  int value;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct doSMLossBack : phast::functor::func_vec<T, policy> {

  _PHAST_METHOD doSMLossBack(const phast::matrix<T>& labelMat_, int maxLabels, int ignore, int ignoreValue) {
    labelMat.link(labelMat_);
    labels = maxLabels;
    hasIgnore = ignore;
    value = ignoreValue;
  }
  
  _PHAST_METHOD T operator()(phast::functor::vector<T>& row) {
	const int label = (int)(*(labelMat.begin_ij() + this->get_index()));
    if (hasIgnore && label == value) {
      this->fill(row.begin(), row.end(), T(0));
      return 0;
    }
    row[label] -= 1;
    return 1;
  }
  
  phast::functor::matrix<T> labelMat;
  int labels;
  int hasIgnore;
  int value;
};


template <typename T, unsigned int policy = phast::get_default_policy()>
struct maxFunc : phast::functor::func_scal_scal<T, policy> {

  _PHAST_METHOD maxFunc() {}
  
  _PHAST_METHOD void operator()(phast::functor::scalar<T>& in,phast::functor::scalar<T>& out) {
    out = smax(in, out);
  }
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct reluFunc : phast::functor::func_scal_scal<T, policy> {

  _PHAST_METHOD reluFunc(T negative) {
    slope = negative;
  }

  _PHAST_METHOD void operator()(phast::functor::scalar<T>& in,phast::functor::scalar<T>& out) {
    out = smax(in, T(0)) + slope * smin(in, T(0));
  }

  T slope;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct reluBackFunc : phast::functor::func_scal_scal_scal<T, policy> {

  _PHAST_METHOD reluBackFunc(T negative) {
    slope = negative;
  }

  _PHAST_METHOD void operator()(phast::functor::scalar<T>& in, phast::functor::scalar<T>& diff, phast::functor::scalar<T>& out) {
    int m0 = in > 0 ? 1 : 0;
    int m1 = in <= 0 ? 1 : 0;
    out = diff * (m0 + slope * m1);
  }

  T slope;
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct func_conv_bp_bias : phast::functor::func_cube_scal<T, policy>
{
  // Inner functor better for GPUs?
  _PHAST_METHOD void operator()(const phast::functor::cube<T>& top_diff, phast::functor::scalar<T>& out) {
	out = this->accumulate(top_diff.begin_ijk(), top_diff.end_ijk());
  }
};

template <typename T, unsigned int policy = phast::get_default_policy()>
struct transposer : phast::functor::func_mat<T, policy>
{
    _PHAST_METHOD transposer(const phast::cube<T>& src, const int num_batches, const int num_filters)
    {
        src_.link(src);
        num_batches_ = num_batches;
        num_filters_ = num_filters;
    }
    _PHAST_METHOD void operator()(phast::functor::matrix<T>& mat)
    {
        const int I = this->get_index() / num_batches_;
        const int J = this->get_index() % num_batches_;

        auto src_mat = *(src_.cbegin_i() + J*num_filters_ + I);
        this->copy(src_mat.cbegin_ij(), src_mat.cend_ij(), mat.begin_ij());
    }

    phast::functor::cube<T> src_;
    int num_batches_;
    int num_filters_;
};


#endif /* __PHAST_FUNCTORS_H__ */

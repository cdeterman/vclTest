
#include "boost/numeric/ublas/matrix.hpp"

#define VIENNACL_WITH_OPENCL 1

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/random.hpp"

#include <Rcpp.h>

using namespace boost::numeric;
using namespace Rcpp;

//' @export
// [[Rcpp::export]]
void test(){
	
	typedef float ScalarType;
	viennacl::tools::uniform_random_numbers<ScalarType> randomNumber;
	int matrix_size = 400;
	
	ublas::matrix<ScalarType> ublas_A(matrix_size, matrix_size);
	ublas::matrix<ScalarType, ublas::column_major> ublas_B(matrix_size, matrix_size);
	ublas::matrix<ScalarType> ublas_C(matrix_size, matrix_size);
	ublas::matrix<ScalarType> ublas_C1(matrix_size, matrix_size);
	
	for (unsigned int i = 0; i < ublas_A.size1(); ++i)
    for (unsigned int j = 0; j < ublas_A.size2(); ++j)
      ublas_A(i,j) = randomNumber();

  for (unsigned int i = 0; i < ublas_B.size1(); ++i)
    for (unsigned int j = 0; j < ublas_B.size2(); ++j)
      ublas_B(i,j) = randomNumber();
      
  /** 
  * Set up some ViennaCL objects. Data initialization will happen later.
  **/
  //viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());  //uncomment this is you wish to use GPUs only
  viennacl::matrix<ScalarType> vcl_A(matrix_size, matrix_size);
  viennacl::matrix<ScalarType, viennacl::column_major> vcl_B(matrix_size, matrix_size);
  viennacl::matrix<ScalarType> vcl_C(matrix_size, matrix_size);

	Rcpp::Rcout << "--- Computing matrix-matrix product using ublas ---" << std::endl;
	ublas_C = ublas::prod(ublas_A, ublas_B);
	
	Rcpp::Rcout << std::endl << "--- Computing matrix-matrix product on each available compute device using ViennaCL ---" << std::endl;
	
  std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();

  for (std::size_t device_id=0; device_id<devices.size(); ++device_id)
  {
    viennacl::ocl::current_context().switch_device(devices[device_id]);
    Rcpp::Rcout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;

    /**
    * Copy the data from the uBLAS objects, compute one matrix-matrix-product as a 'warm up', then take timings:
    **/
    viennacl::copy(ublas_A, vcl_A);
    viennacl::copy(ublas_B, vcl_B);
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    viennacl::backend::finish();
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    viennacl::backend::finish();

    /**
    * Verify the result
    **/
    viennacl::copy(vcl_C, ublas_C1);

    std::cout << " - Checking result... ";
    bool check_ok = true;
    for (std::size_t i = 0; i < ublas_A.size1(); ++i)
    {
      for (std::size_t j = 0; j < ublas_A.size2(); ++j)
      {
        if ( std::fabs(ublas_C1(i,j) - ublas_C(i,j)) / ublas_C(i,j) > 1e-4 )
        {
          check_ok = false;
          break;
        }
      }
      if (!check_ok)
        break;
    }
    if (check_ok)
      Rcpp::Rcout << "[OK]" << std::endl << std::endl;
    else
      Rcpp::Rcout << "[FAILED]" << std::endl << std::endl;

  }
	
	Rcpp::Rcout << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

}


//' @export
// [[Rcpp::export]]
void test_double(){
	
	typedef double ScalarType;
	viennacl::tools::uniform_random_numbers<ScalarType> randomNumber;
	int matrix_size = 400;
	
	ublas::matrix<ScalarType> ublas_A(matrix_size, matrix_size);
	ublas::matrix<ScalarType, ublas::column_major> ublas_B(matrix_size, matrix_size);
	ublas::matrix<ScalarType> ublas_C(matrix_size, matrix_size);
	ublas::matrix<ScalarType> ublas_C1(matrix_size, matrix_size);
	
	for (unsigned int i = 0; i < ublas_A.size1(); ++i)
    for (unsigned int j = 0; j < ublas_A.size2(); ++j)
      ublas_A(i,j) = randomNumber();

  for (unsigned int i = 0; i < ublas_B.size1(); ++i)
    for (unsigned int j = 0; j < ublas_B.size2(); ++j)
      ublas_B(i,j) = randomNumber();
      
  /** 
  * Set up some ViennaCL objects. Data initialization will happen later.
  **/
  //viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());  //uncomment this is you wish to use GPUs only
  viennacl::matrix<ScalarType> vcl_A(matrix_size, matrix_size);
  viennacl::matrix<ScalarType, viennacl::column_major> vcl_B(matrix_size, matrix_size);
  viennacl::matrix<ScalarType> vcl_C(matrix_size, matrix_size);

	Rcpp::Rcout << "--- Computing matrix-matrix product using ublas ---" << std::endl;
	ublas_C = ublas::prod(ublas_A, ublas_B);
	
	Rcpp::Rcout << std::endl << "--- Computing matrix-matrix product on each available compute device using ViennaCL ---" << std::endl;
	
  std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();

  for (std::size_t device_id=0; device_id<devices.size(); ++device_id)
  {
    viennacl::ocl::current_context().switch_device(devices[device_id]);
    Rcpp::Rcout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;

    /**
    * Copy the data from the uBLAS objects, compute one matrix-matrix-product as a 'warm up', then take timings:
    **/
    viennacl::copy(ublas_A, vcl_A);
    viennacl::copy(ublas_B, vcl_B);
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    viennacl::backend::finish();
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    viennacl::backend::finish();

    /**
    * Verify the result
    **/
    viennacl::copy(vcl_C, ublas_C1);

    std::cout << " - Checking result... ";
    bool check_ok = true;
    for (std::size_t i = 0; i < ublas_A.size1(); ++i)
    {
      for (std::size_t j = 0; j < ublas_A.size2(); ++j)
      {
        if ( std::fabs(ublas_C1(i,j) - ublas_C(i,j)) / ublas_C(i,j) > 1e-4 )
        {
          check_ok = false;
          break;
        }
      }
      if (!check_ok)
        break;
    }
    if (check_ok)
      Rcpp::Rcout << "[OK]" << std::endl << std::endl;
    else
      Rcpp::Rcout << "[FAILED]" << std::endl << std::endl;

  }
	
	Rcpp::Rcout << "!!!! DOUBLE TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

}




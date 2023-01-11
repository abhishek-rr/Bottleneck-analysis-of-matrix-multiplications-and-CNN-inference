#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void initiaze_matrices(double** a, double** b, double** c, int P, int N, int K)
{
  int i,j;
  int Cp = P-K+1;
  int Cn = N-K+1;
  #pragma omp parallel for private(i,j) shared(a)
    for (i=0; i<P; i++)
      for (j=0; j<N ; j++)
        a[i][j] = i+j;

  #pragma omp parallel for private(i,j) shared(b)
  for (i=0; i< K; i++)
    for(j=0; j<K; j++)
      b[i][j] = i*j * pow(-1, j+1);
  //;// to test negative numbers (for ReLU):
  // printf("Cp = %d\n", Cp);
  // printf("Cn = %d\n", Cn);
  #pragma omp parallel for private(i,j) shared(c)
  for (i=0; i<Cp; i++)
    for(j=0; j<Cn; j++)
      c[i][j] = 0;
}

void matmul(double** a, double** b, double** c, int P, int N, int K)
{
  int i,j,kh, kv;
  int Cp = P-K+1;
  int Cn = N-K+1;
  #pragma omp parallel for private(i,j,kh, kv) collapse(4) shared(a,b,c)
  for(kh = 0; kh < Cp; kh++)
    for(kv = 0; kv < Cn; kv++)
      for (i=0; i<K; i++)
        for(j=0; j<K; j++) 
          c[kh][kv] += a[kh+i][kv+j] * b[i][j];
  return;
}

void ReLU(double** c, int P, int N)
{
  {
    int i,j;
    #pragma omp parallel for private(i,j) collapse(2)
    for (i=0; i<P; i++)
      for(j=0; j<N; j++)
        c[i][j] = c[i][j]>0 ? c[i][j] : 0;
  }
  return;
}

void print_matrix(double** c, int P, int N)
{
  int i,j;
  for (i=0; i<P; i++)
  {
    for(j=0; j<N; j++)
      printf("c[%d][%d] = %f",i,j,c[i][j]);
    printf("\n");
  }
  return;
}

int main(int argc, char *argv[])
{
    int P = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int num_threads = atoi(argv[4]);
    int i;
    double start = omp_get_wtime();
    omp_set_num_threads(num_threads);
    

    double** a;
    double** b;
    double** c;

    int CP = P-K+1;
    int CN = N-K+1;

    a = (double **) malloc((P)*sizeof(double*));
    for (i = 0; i < P; i++)
        a[i] = (double*)malloc(N * sizeof(double));
    
    b = (double **) malloc((K)*sizeof(double*));
    for (i = 0; i < K; i++)
        b[i] = (double*)malloc(K * sizeof(double));
    
    c = (double **) malloc((CP)*sizeof(double*));
    for (i = 0; i < CP; i++)
        c[i] = (double*)malloc(CN * sizeof(double));

    double startinit = omp_get_wtime();
    initiaze_matrices(a, b, c, P, N, K);
    double endinit = omp_get_wtime();

    double startmul = omp_get_wtime();
    matmul(a, b, c, P, N, K);
    double endmul = omp_get_wtime();

    double startrelu = omp_get_wtime();
    ReLU(c, CP, CN);
    double endrelu = omp_get_wtime();

    double end = omp_get_wtime();
    double execTime = end- start;

    printf("initiaze_matrices(): %f\n",endinit-startinit);
    printf("percentage: %f\n\n",(endinit-startinit)/execTime *100);

    printf("matmul(): %f\n",endmul - startmul);
    printf("percentage: %f\n\n",(endmul - startmul)/execTime *100);

    printf("ReLU(): %f\n",endrelu - startrelu);
    printf("percentage: %f\n\n",(endrelu - startrelu)/execTime *100);
    
    printf("execution time: %f\n",execTime);
    printf("\n");
    
    // print_matrix(c, P-K+1, N-K+1);
    double Tseq = 0.194529;
    printf("Speedup: %f\n", Tseq/execTime );
    
    free(a);
    free(b);
    free(c);
    return 0;
}

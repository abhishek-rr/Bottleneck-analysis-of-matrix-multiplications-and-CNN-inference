#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void initiaze_matrices(double** a, double** b, double** c, int N)
{
  int i,j;
  #pragma omp parallel for private(i,j) shared(a)
    for (i=0; i<N; i++)
      for (j=0; j<N ; j++)
        a[i][j] = i+j;

  #pragma omp parallel for private(i,j) shared(b)
  for (i=0; i< N; i++)
    for(j=0; j<N; j++)
      b[i][j] = i*j;
  
  #pragma omp parallel for private(i,j) shared(c)
  for (i=0; i<N; i++)
    for(j=0; j<N; j++)
      c[i][j] = 0;
}

void matmul(double** a, double** b, double** c, int N)
{
  int i,j;
  #pragma omp parallel for private(i,j) collapse(2) shared(a,b,c)
  for (i=0; i<N; i++)
    for(j=0; j<N; j++)
      c[i][j] = a[i][j] * b[i][j];
  return;
}

void print_matrix(double** c, int N)
{
  int i,j;
  for (i=0; i<N; i++)
  {
    for(j=0; j<N; j++)
      printf("c[%d][%d] = %f",i,j,c[i][j]);
    printf("\n");
  }
  return;
}

int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int i;
    double start = omp_get_wtime();
    omp_set_num_threads(num_threads);
    
    double** a;
    double** b;
    double** c;
    a = (double **) malloc((N)*sizeof(double*));
    for (i = 0; i < N; i++)
        a[i] = (double*)malloc(N * sizeof(double));
    b = (double **) malloc((N)*sizeof(double*));
    for (i = 0; i < N; i++)
        b[i] = (double*)malloc(N * sizeof(double));
    c = (double **) malloc((N)*sizeof(double*));
    for (i = 0; i < N; i++)
        c[i] = (double*)malloc(N * sizeof(double));

    
    
    double startinit = omp_get_wtime();
    initiaze_matrices(a, b, c, N);
    double endinit = omp_get_wtime();

    double startmul = omp_get_wtime();
    matmul(a, b, c, N);
    double endmul = omp_get_wtime();

    double end = omp_get_wtime();
    double execTime = end- start;

    printf("initiaze_matrices(): %f\n",endinit-startinit);
    printf("percentage: %f\n\n",(endinit-startinit)/execTime *100);

    printf("matmul(): %f\n",endmul - startmul);
    printf("percentage: %f\n\n",(endmul - startmul)/execTime *100);
    
    printf("execution time: %f\n",execTime);
    printf("\n");

    double Tseq = 7.153127;
    printf("Speedup: %f\n", Tseq/execTime );

    free(a);
    free(b);
    free(c);
    return 0;
}

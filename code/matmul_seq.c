#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void initiaze_matrices(double** a, double** b, double** c, int N)
{
  int i,j;  
  for (i=0; i<N; i++)
    for (j=0; j<N ; j++)
      a[i][j] = i+j;
  
  for (i=0; i< N; i++)
    for(j=0; j<N; j++)
      b[i][j] = i*j;
  
  for (i=0; i<N; i++)
    for(j=0; j<N; j++)
      c[i][j] = 0;
}

void matmul(double** a, double** b, double** c, int N)
{
  int i,j,k;
  for (i=0; i<N; i++)
    for(j=0; j<N; j++)
      for(k=0; k<N; k++)
        c[i][j] += a[i][k] * b[k][j];
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

    
    
    initiaze_matrices(a, b, c, N);
    matmul(a, b, c, N);

    double end = omp_get_wtime();
    double execTime = end- start;
        
    printf("execution time: %f\n",execTime);
    free(a);
    free(b);
    free(c);
    return 0;
}

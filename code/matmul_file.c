#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void initiaze_matrices(double** a, double** b, double** c, int N, double* nums)
{
  int i,j;
  #pragma omp parallel for private(i,j) shared(a)
    for (i=0; i<N; i++)
      for (j=0; j<N ; j++)
        a[i][j] = nums[N*i+j];

  #pragma omp parallel for private(i,j) shared(b)
  for (i=0; i< N; i++)
    for(j=0; j<N; j++)
      b[i][j] = nums[(N * N)- 1  + (N*i+j)];
  
  #pragma omp parallel for private(i,j) shared(c)
  for (i=0; i<N; i++)
    for(j=0; j<N; j++)
      c[i][j] = nums[(2* N * N)- 1 + (N * i + j)];
}

void matmul(double** a, double** b, double** c, int N)
{
  int i,j,k;
  #pragma omp parallel for private(i,j,k) shared(a,b,c)
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
    
    //inititalize matrices
    
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

    //reading from file
    double* nums;
    FILE *file;
    char tok[50];
    file = fopen("nums.txt", "r");
    double totalnums = 3 * N * N;
    nums = (double *) malloc((totalnums)*sizeof(double*));
    for (i = 0; i < totalnums; i++)
    {
        fscanf(file, "%s", tok);
        nums[i] = atof(tok);
    }
    fclose(file);

    double startinit = omp_get_wtime();
    initiaze_matrices(a, b, c, N, nums);
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

    double Tseq = 28.232750;
    printf("Speedup: %f\n", Tseq/execTime );

    free(a);
    free(b);
    free(c);
    return 0;
}

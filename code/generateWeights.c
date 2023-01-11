#include <stdio.h>
#include <stdlib.h>

void initiaze_matrices(int P, int N, int K, FILE* fptr)
{
  int i,j;
  for (i=0; i<P; i++)
    for (j=0; j<N ; j++)
      fprintf(fptr,"%d\n", i+j);

  for (i=0; i< K; i++)
    for(j=0; j<K; j++)
      fprintf(fptr,"%d\n", i*j);// to test negative numbers (for ReLU): * pow(-1, j+1);
}

int main(int argc, char *argv[])
{
  int P = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  FILE *fptr;
  
  fptr = fopen("weights.txt","w");
  
  initiaze_matrices(P, N, K, fptr);
  fclose(fptr);

  return 0;
}
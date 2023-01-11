#include <stdio.h>
#include <stdlib.h>


void initiaze_matrices(int N, FILE* fptr)
{
  int i,j;
  for (i=0; i<N; i++)
    for (j=0; j<N ; j++)
      fprintf(fptr,"%d\n", i+j);

  for (i=0; i< N; i++)
    for(j=0; j<N; j++)
      fprintf(fptr,"%d\n", i*j);

  for (i=0; i<N; i++)
    for(j=0; j<N; j++)
      fprintf(fptr,"%d\n", 0);
}

int main(int argc, char *argv[])
{
  int N = atoi(argv[1]);
  FILE *fptr;
  
  fptr = fopen("nums.txt","w");
  
  initiaze_matrices(N, fptr);
  fclose(fptr);

  return 0;
}
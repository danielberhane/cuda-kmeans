/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_main.cu                                            */
/*   Description:  Main driver for CUDA k-means clustering                 */
/*                                                                         */
/*   Author:  Daniel Berhane Araya                                         */
/*   Copyright (c) 2016 Daniel Berhane Araya                               */
/*                                                                         */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// MIT License - see LICENSE file for details

#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "kmeans.h"

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {

           int     is_output_timing;
           int     numClusters, numCoords, numObjs;
           int    *membership;    /* [numObjs] */
           char   *filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;
           double  timing, io_timing, clustering_timing;
           int     loop_iterations = 500;
           int a;

  if (argc < 3) {
    printf("Too few arguments\n");
    printf("Command needs to be (./km cluster# dimension datasize filename)\n");
    return 0;
  }

  /* cluster size, dimension, data-size and counter to keep track the number of iterations
     before converging
  */
  numClusters = atoi(argv[1]);
  numCoords = atoi(argv[2]);
  numObjs = atoi(argv[3]);
  threshold = atof(argv[4]);
  
  filename = argv[5];
  threshold = (float)threshold;
  is_output_timing = 1;

  if (is_output_timing) io_timing = wtime();

  objects = file_read(filename , &numObjs, &numCoords);
   if (objects == NULL) exit(1);

 if (is_output_timing) {
        timing            = wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }
    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

    a = cuda_kmeans(objects, numCoords, numObjs, numClusters, threshold, membership, &loop_iterations, &clusters);

    if (a);

    free(objects[0]);
    free(objects);

    if (is_output_timing) {
        timing            = wtime();
        clustering_timing = timing - clustering_timing;
    }

file_write(filename, numClusters, numObjs, numCoords, clusters,
               membership);

    free(membership);
    free(clusters[0]);
    free(clusters);

if (is_output_timing) {
        io_timing += wtime() - timing;
	
        printf("\n*** Kmeans (CUDA version) Output ***\n");
	printf("\n");
        printf("Input file:     %s\n", filename);
        printf("numObjs       = %d\n", numObjs);
        printf("numCoords     = %d\n", numCoords);
        printf("numClusters   = %d\n", numClusters);
        printf("threshold     = %0.2f \n", threshold);

       // printf("Loop iterations    = %d\n", loop_iterations);

        printf("I/O time           = %10.4f sec\n", io_timing);
        printf("Computation timing = %10.4f sec\n", clustering_timing);
	printf("\n");
	}

    return(0);


}







/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_kmeans.cu                                          */
/*   Description:  CUDA kernels for k-means clustering                     */
/*                                                                         */
/*   Author:  Daniel Berhane Araya                                         */
/*   Copyright (c) 2016 Daniel Berhane Araya                               */
/*                                                                         */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// MIT License - see LICENSE file for details

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "kmeans.h"

static int nextPowerOfTwo (int n) {
  n--;

  n = n >> 1 | n;
  n = n >> 2 | n;
  n = n >> 4 | n;
  n = n >> 8 | n;
  n = n >> 16 | n;
  // n = n >> 32 | n;  For 64-bit ints

  return ++n;
}

__host__ __device__ static float euclid_dist (int numCoords,
					      int numObjs,
					      int numClusters,
					      float *objects, // [numCoords][numObjs]
					      float *clusters, // [numCoords][numClusters]
					      int objectId,
					      int clusterId)
{
  int i;
  float ans=0.0;

  for (i=0; i<numCoords; i++) {
    ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
      (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
  }

  return (ans);
}

/*
  key assumption: number of threadBlocks is power of 2
  
 */


//...................................... Find the nearest cluster kernel ...............................

__global__ static void find_nearest_cluster(int numCoords,
					    int numObjs,
					    int numClusters,
					    float *objects, // [numCoords][numObjs]
					    float *deviceClusters, //[numCoords][numClusters]
					    int *membership, // [numObjs]
					    int *intermediates,
					    int *sumClusters,
					    float *sumCoordinates)

{

  extern __shared__ char sharedMemory[];

  float *s_objects = (float *) sharedMemory;
  float *s_clusters = (float *) (s_objects + (blockDim.x * numCoords));
  int *s_sum_clusters = (int *) (s_clusters + (numClusters * numCoords));
  float *s_sum_coords = (float *) (s_sum_clusters + numClusters);
  unsigned char *s_memb_changed = (unsigned char *) (s_sum_coords + numClusters * numCoords);
  unsigned char *s_membership = (unsigned char *) (s_memb_changed + blockDim.x);

  s_memb_changed[threadIdx.x] = 0;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

   
  /*
     initialize so that the extra space introduced to make the kernel reduce_coord_clusters reduce
    2 * numThreadBlocks without remainder is set to zero

  */ 

  if (idx < ((gridDim.x + 2 * blockDim.x - (gridDim.x % (2 * blockDim.x))) * numClusters)) {
    sumClusters[idx] = 0;
  }	 

  if (idx < ((gridDim.x + 2 * blockDim.x - (gridDim.x % (2 * blockDim.x))) * numClusters * numCoords)) {
    sumCoordinates[idx] = 0;
  }

  /*
    Load data if numCoords == 1 the next logical expression becomes idx < numObjs. Objects stored in
    x0x1x2 y0y1y2 ... for mat [numCoords][numObjs]
  */

  if (idx < numObjs*numCoords) {
    for (int i=0; i<numCoords; i++) {
      s_objects[blockDim.x * i + threadIdx.x] = objects[numObjs * i + idx];
      // printf("shared %f \n ", objects[numObjs*i + idx]);
    }
  }
  __syncthreads(); // make sure all data is loaded - MOVED OUTSIDE IF
             
  /*
   Load clusters
  */
   
  if (threadIdx.x < numClusters) {
    for (int i=0; i<numCoords; i++) {
      s_clusters[numClusters * i + threadIdx.x] = deviceClusters[numClusters * i + threadIdx.x];
    }
  }
  __syncthreads(); // make sure all clusters are loaded - MOVED OUTSIDE IF

  /*
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {

    for (int x =0; x<numCoords*numClusters; x++)
      printf("Clusters %f \n ", s_clusters[x]);
  }
  */

  if (idx < numObjs) {
       
    int index,i;
    float dist, min_dist;
       
    // find the cluster with the minimum distance
    
    index = 0;
    min_dist = euclid_dist(numCoords, blockDim.x, numClusters, s_objects, s_clusters, threadIdx.x, 0);
    
    for (i=1; i<numClusters; i++) {
      dist = euclid_dist(numCoords, blockDim.x, numClusters, s_objects, s_clusters, threadIdx.x, i);
   
      if (dist < min_dist) {
	min_dist = dist;
	index = i;
      }
    }

    if (idx < numObjs) {
      if (membership[idx] != index) {
	s_memb_changed[threadIdx.x] = (unsigned char)1;
	s_membership[threadIdx.x] = index;
	membership[idx] = index;
      }
      //  printf("membership %d block %d thread %d\n", membership[idx], blockIdx.x, threadIdx.x);
    }
    //  printf("Shared mem. memb changed %d block %d thread %d\n ", s_memb_changed[threadIdx.x], blockIdx.x, threadIdx.x);
  }
  __syncthreads(); // for s_memb_changed - MOVED OUTSIDE IF
    
  for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (threadIdx.x < s) 
      s_memb_changed[threadIdx.x] += s_memb_changed[threadIdx.x + s];
      __syncthreads();
      

      if (threadIdx.x < 32) {
      	 s_memb_changed[threadIdx.x] += s_memb_changed[threadIdx.x + 32];
	 s_memb_changed[threadIdx.x] += s_memb_changed[threadIdx.x + 16];
	 s_memb_changed[threadIdx.x] += s_memb_changed[threadIdx.x + 8];
	 s_memb_changed[threadIdx.x] += s_memb_changed[threadIdx.x + 4];
	 s_memb_changed[threadIdx.x] += s_memb_changed[threadIdx.x + 2];
	 s_memb_changed[threadIdx.x] += s_memb_changed[threadIdx.x + 1];
      
      }
      
    }

  if (threadIdx.x == 0) {
    intermediates[blockIdx.x] = s_memb_changed[0];
    //  printf("Intermediates %d block %d \n", intermediates[blockIdx.x], blockIdx.x);

  }

  /*
     initialize both s_sum_clusters and s_sum _coords, both shared memories are used to collect the sums of clusters and
     coordinates per each block which will later be saved to the global variables sumClusters and sumCoordinates
     
  */

  if (threadIdx.x < numClusters)
    s_sum_clusters[threadIdx.x] = 0;

  if (threadIdx.x < numCoords) {
    for (int i=0; i<numClusters; i++)
      s_sum_coords[numCoords * i + threadIdx.x] = (float) 0;
  }

 
  // update the shared memories of sums of clusters and coordinates by going through the shared memory s_membership 

  if (threadIdx.x < numClusters) {

    for (int i=0; i<blockDim.x; i++) {
      if (s_membership[i] == threadIdx.x) {
	++s_sum_clusters[threadIdx.x];
	      
	for (int j=0; j<numCoords; j++) {
	  s_sum_coords[j+threadIdx.x*numCoords] += s_objects[i+blockDim.x*j];
	  
	}
      }
    }
  }
  __syncthreads(); // make sure that all threads sum the clusters and coordinates before saving it to global memory

  // save the sums to global variables

  if (threadIdx.x < numClusters) {
    int idn = threadIdx.x + numClusters*blockIdx.x;
    sumClusters[idn] = s_sum_clusters[threadIdx.x];
    // printf("Sum clusters %d \n ", sumClusters[idn]);		          

  }
    
  if (threadIdx.x < numClusters*numCoords) {
    int idn = threadIdx.x + numClusters*numCoords*blockIdx.x;
    sumCoordinates[idn] = s_sum_coords[threadIdx.x];
    //  printf("Sum Coords %.2f \n ", sumCoordinates[idn]);
  }
   
}





// ...................................... reduce_coord_clusters kernel.............................................


/*
  reduce_coord_clusters kernel -  each block sums the corresponding sums of clusters and coordinates. eg. block 0
  finds the sum of the number of objects in cluster 0, and sum of coordinates of objects in cluster 0,
  block 1 takes care of cluster 1 ... and so on.


 */

__global__ static void reduce_coord_clusters(int numCoords,
					     int numClusters,
					     int *sum_cluster, // [numClusters][gridSize]
					     float *sum_coords, //[ numCoords][numClusters][gridSize]
					     int *dev_cluster_sums, // [numClusters]
					     float *dev_coord_sums, // [numCoords][numClusters]
					     int *size_cluster_array // the size of the global variable sum_cluster
					     )

{

  extern __shared__ char sharedMemory[];

  int *s_clusters = (int *) sharedMemory;
  float *s_coords = (float *) (s_clusters + 2 * blockDim.x);   // size 2 * blockDim.x
  int *s_total_clusters = (int *) (s_coords + 2 * blockDim.x);  // size one [a block handles only one cluster]
  float *s_total_coords = (float *) (s_total_clusters + 1);  // size numCoords

  // initialize sum of cluster and sum of coordinates
  s_total_clusters[0] =  0;
  
  for (unsigned int i=0; i<numCoords; i++)
    s_total_coords[i] = 0;
  
  
    
  // printf("Size reduce %d \n ", *size_cluster_array);
    

  
  // Outer most loop takes care of how many numbers of 2 * threadblocks are taken to the shared memory for reduction

  for (int j=0; j< (*size_cluster_array)/(2*blockDim.x); j++) {   
      
    // Load cluster sums for reduction
    if (blockIdx.x < numClusters) {
	
      for (int i=0; i<2; i++) {
	s_clusters[threadIdx.x + i * blockDim.x] = sum_cluster[numClusters* ((i*blockDim.x) + (threadIdx.x) + (2 * blockDim.x * j)) + blockIdx.x]; 
	// printf("s_clusters %d  j %d block %d thread %d \n ", s_clusters[threadIdx.x + i * blockDim.x], j, blockIdx.x, threadIdx.x);
      }
      __syncthreads();  // make sure all threads have taken the data
    }
      
    /*
      the loaded clusters are power of two because the size if 2 * blockDim and blockDim is power of 2 already
      so that parallel reduction can be done  
    */

    for (unsigned int s = blockDim.x; s>32; s >>=1) {
      if (threadIdx.x < s) 
	s_clusters[threadIdx.x] += s_clusters[threadIdx.x + s];
	
      __syncthreads(); // for reduction
    }

    if (threadIdx.x < 32) {
       s_clusters[threadIdx.x] += s_clusters[threadIdx.x + 32];
       s_clusters[threadIdx.x] += s_clusters[threadIdx.x + 16];
       s_clusters[threadIdx.x] += s_clusters[threadIdx.x + 8];
       s_clusters[threadIdx.x] += s_clusters[threadIdx.x + 4];
       s_clusters[threadIdx.x] += s_clusters[threadIdx.x + 2];
       s_clusters[threadIdx.x] += s_clusters[threadIdx.x + 1];
    
    }




    
    if (threadIdx.x == 0) {

      s_total_clusters[0] += s_clusters[0];
      
    }
      
    // This loads each coordinate for reduction
      
    if (blockIdx.x < numClusters) {

      for (int k=0; k<numCoords; k++) {
	  
	// Loading coordinates
	for (int c=0; c<2; c++) {
	  s_coords[threadIdx.x + c * blockDim.x] = sum_coords[numCoords * numClusters * ((c * blockDim.x) + (threadIdx.x) + (j * 2 * blockDim.x)) + blockIdx.x * numCoords + k];
	  // printf("S_coord %f block %d thread %d \n ", sum_coords[threadIdx.x + c*blockDim.x], blockIdx.x, threadIdx.x);
	}
	__syncthreads();   // make sure all coordinates are loaded


	/*
	
	if ((threadIdx.x == 0) && (blockIdx.x < numClusters)) {
	  for (int x = 0; x < 2*blockDim.x; x++) {
	    //	  printf("Sum coords %f block %d thread %d \n ", s_coords [x], blockIdx.x, threadIdx.x);
	  }
	}
	    
	if ((blockIdx.x < numClusters) && (threadIdx.x == 0)) {
	  for (int x = 0; x<2*blockDim.x; x++) {
	    //	printf("sum %f j %d block %d thread %d  \n ", s_coords[x], j, blockIdx.x, threadIdx.x);

	  }
	}


	*/
	
	// Reduce the coordinate and add the sum to the correstponding array
	  
	for (unsigned int s = blockDim.x; s>32; s >>=1) {
	  if (threadIdx.x < s) 
	    s_coords[threadIdx.x] += s_coords[threadIdx.x + s];
	     
	  __syncthreads();
	}

	if (threadIdx.x < 32) {
	       s_coords[threadIdx.x] += s_coords[threadIdx.x + 32];
	           s_coords[threadIdx.x] += s_coords[threadIdx.x + 16];
		       s_coords[threadIdx.x] += s_coords[threadIdx.x + 8];
		           s_coords[threadIdx.x] += s_coords[threadIdx.x + 4];
			       s_coords[threadIdx.x] += s_coords[threadIdx.x + 2];
			           s_coords[threadIdx.x] += s_coords[threadIdx.x + 1];
	


	}
	



	
	  
	if (threadIdx.x == 0) {
	  s_total_coords[k] += s_coords[0];
	  //  printf("Total %f \n ", s_total_coords[k]);
	}
	
	__syncthreads();
      }
    }

  } // end of outer most for loop



  // finally for the thread block save the sumOfClusters from shared to global memory
    
  if (blockIdx.x < numClusters) {
    dev_cluster_sums [blockIdx.x] = *s_total_clusters;
	   
  }

  // printf("Sum of clusters %d \n ", dev_cluster_sums[blockIdx.x]);

  if ((threadIdx.x < numCoords) && (blockIdx.x < numClusters)) {

    dev_coord_sums[blockIdx.x * numCoords + threadIdx.x] =  s_total_coords[threadIdx.x];
    //printf("Coords %f \n ", dev_coord_sums[idx]);

  }
 
} // outer most loop


/*......................... reduce_cluster_changed kernel ..............................................

  Given the limit of the size of threadBlocks per block is 1024, if the data is larger than 100,000,
  this kernel will have an error because when the reduction is done, it is done using one block. If
  the number of objects is so large that the previous find_nearest_cluster kernel produces an intermediate
  size with gridDim size greater than 1024, this will give an error here, because, that gridDim size in
  find_nearest_cluster will be the blockDim in this kernel
  
 */

__global__ static void reduce_cluster_changed (int *deviceIntermediates,
					       int numIntermediates,
					       int numIntermediates2)
{

  extern __shared__ unsigned int intermediates[];

  
  intermediates[threadIdx.x] = (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

  __syncthreads();

  // numIntermediates2 have to be a power of two

  for (unsigned int s = numIntermediates2/2; s>32; s>>=1) {
    if (threadIdx.x < s) 
      intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
    
    __syncthreads();
  }

  if (threadIdx.x < 32) {
        intermediates[threadIdx.x] += intermediates[threadIdx.x + 32];
	   intermediates[threadIdx.x] += intermediates[threadIdx.x + 16];
	      intermediates[threadIdx.x] += intermediates[threadIdx.x + 8];
	         intermediates[threadIdx.x] += intermediates[threadIdx.x + 4];
		    intermediates[threadIdx.x] += intermediates[threadIdx.x + 2];
		       intermediates[threadIdx.x] += intermediates[threadIdx.x + 1];
		       

  }









  
  if (threadIdx.x == 0) {
    deviceIntermediates[0] = intermediates[0];
    printf("\n");
    // printf("Sum of Changed clusters %d\n", deviceIntermediates[0]);
  }

}

int cuda_kmeans (float **objects,  // in : [numObjs][numCoords]
		 int numCoords,
		 int numObjs,
		 int numClusters,
		 float threshold,
		 int *membership,
		 int *loop_iterations,
		 float ***clusters_out,
		 kmeans_perf_t *perf)   /* may be NULL */

{
  int i, j, loop=0;
  float delta;
  float **dimObjects;
  float **dimClusters;
  //  float **clusters;
  // float changedClusters;
  //float **newCluster;

  float *deviceObjects;
  float *deviceClusters;
  int *deviceMembership;
  int *deviceIntermediates;
  // used in find_cluser kernel
  int *dev_sum_clusters; // sums the number of cluster in each thread block 
  float *dev_sum_coords;  // sums the value of the coordinates of each thread block



  // used in reducer kernel

  int *final_sum_clusters;
  float *final_sum_coords;
        
  // Copy objects given in [numObjs][numCoords] layout to new
  // [numCoords][numObjs] layout

  malloc2D(dimObjects, numCoords, numObjs, float);

  for (i=0; i<numCoords; i++) {
    for (j=0; j<numObjs; j++) {
      dimObjects[i][j] = objects[j][i];
      //  printf("data %.2f\n ", dimObjects[i][j]);
    }
  }

  malloc2D (dimClusters, numCoords, numClusters, float);
  for (i=0; i<numCoords; i++) {
    for (j=0; j<numClusters; j++) {
      dimClusters[i][j] = dimObjects[i][j];
      // printf("clusters %.2f   ", dimClusters[i][j]);
    }
  }

  // initialize membership array
  for (i=0; i<numObjs; i++) membership[i] = -1;


  const unsigned int numThreadsPerClusterBlock = 128;  // note that number of threadBlocks should be power of two
  const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

   
  // size of shared memory for find_cluster kernel

  const unsigned int clusterBlockSharedDataSize = (numThreadsPerClusterBlock * numCoords * sizeof(float)) + (numClusters * numCoords * sizeof(float))  + (numClusters * sizeof (int)) +
    (numClusters * numCoords * sizeof(float))+ (numThreadsPerClusterBlock * sizeof(unsigned char)) + (numThreadsPerClusterBlock * sizeof(unsigned char));
 
 
  // shared memory for reduce_coord_clusters_kernel
  const unsigned int shared_reducer = (2 * numThreadsPerClusterBlock * sizeof(int)) + (2 * numThreadsPerClusterBlock * sizeof(float)) + (sizeof (int)) + (numCoords * sizeof(float));


  // Check just in case we run out of shared memory size
  
  cudaDeviceProp deviceProp;
  int deviceNum;
  cudaGetDevice (&deviceNum);
  cudaGetDeviceProperties(&deviceProp, deviceNum);

  // printf("Asked for %u\n", clusterBlockSharedDataSize);
  // printf("Available %lu\n", deviceProp.sharedMemPerBlock);

  const unsigned int numReductionThreads =
    nextPowerOfTwo(numClusterBlocks);
  const unsigned int reductionBlockSharedDataSize =
    numReductionThreads * sizeof(unsigned int);

  int reducer_size = numClusterBlocks + 2 * numThreadsPerClusterBlock - (numClusterBlocks % (2 * numThreadsPerClusterBlock));

  // find_nearest_cluster kernel variables

  checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
  checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
  checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
  checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

  checkCuda(cudaMalloc(&dev_sum_clusters, numClusters * reducer_size * sizeof(int)));
  checkCuda(cudaMalloc(&dev_sum_coords, numClusters * numCoords * reducer_size * sizeof(float)));


  checkCuda(cudaMemcpy(deviceObjects, dimObjects[0], numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(deviceMembership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice));
  

  // reducer_coord_cluster kernel variables

  int *dev_num_reduce_clusters;
    
  checkCuda(cudaMalloc(&final_sum_clusters, numClusters*sizeof(int)));
  checkCuda(cudaMalloc(&final_sum_coords, numCoords*numClusters*sizeof(float)));
  checkCuda(cudaMalloc(&dev_num_reduce_clusters, sizeof(int)));
  checkCuda(cudaMemcpy(dev_num_reduce_clusters, &reducer_size, sizeof(int), cudaMemcpyHostToDevice));
    
  
  /*
    Find nearest cluster, reduce the sum, find new centroid ... loop these three until a threshold value is reached
    the threshold is a value delta = numChanged/numObjects. The loop will run until threshold is less than delta
  */

  /* ---- benchmark instrumentation ------------------------------------------
     All three kernels are launched on the default stream, so they serialize
     against each other without an explicit cudaDeviceSynchronize().  The two
     blocking cudaMemcpy calls in the loop body provide the synchronization
     points that surface any execution error.  Per-kernel cudaEvent timing is
     compiled in only for the breakdown build, because the syncs it requires
     serialize the pipeline and inflate the total.
     ------------------------------------------------------------------------*/
  double t_transfer = 0.0, t_xfer0;
  float  acc_find = 0.0f, acc_reduce_coord = 0.0f, acc_reduce_changed = 0.0f;

#ifdef BENCH_KERNEL_BREAKDOWN
  cudaEvent_t k_start, k_stop;
  float k_ms = 0.0f;
  checkCuda(cudaEventCreate(&k_start));
  checkCuda(cudaEventCreate(&k_stop));
  #define KERNEL_TIME_BEGIN()   cudaEventRecord(k_start)
  #define KERNEL_TIME_END(acc)  do {                              \
      cudaEventRecord(k_stop);                                    \
      cudaEventSynchronize(k_stop);                               \
      cudaEventElapsedTime(&k_ms, k_start, k_stop);               \
      (acc) += k_ms;                                              \
  } while (0)
#else
  #define KERNEL_TIME_BEGIN()   do { } while (0)
  #define KERNEL_TIME_END(acc)  do { (void)(acc); } while (0)
#endif

  /* Hoisted out of the convergence loop: these were previously malloc'd on
     every iteration and never freed.                                        */
  int   *cluster_count = (int *)   malloc(numClusters * sizeof(int));
  float *coord_sum     = (float *) malloc(numCoords * numClusters * sizeof(float));
  assert(cluster_count != NULL && coord_sum != NULL);

  const double t_loop_start = wtime();

  do {

    loop++;

    t_xfer0 = wtime();
    checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    t_transfer += wtime() - t_xfer0;

    KERNEL_TIME_BEGIN();
    find_nearest_cluster<<<numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize>>>
      (numCoords, numObjs, numClusters, deviceObjects, deviceClusters, deviceMembership, deviceIntermediates, dev_sum_clusters, dev_sum_coords);
    KERNEL_TIME_END(acc_find);
    checkLastCudaError();   /* launch errors; execution errors surface at the memcpy below */

    KERNEL_TIME_BEGIN();
    reduce_cluster_changed<<<1, numReductionThreads, reductionBlockSharedDataSize >>>
      (deviceIntermediates, numClusterBlocks, numReductionThreads);
    KERNEL_TIME_END(acc_reduce_changed);
    checkLastCudaError();

    int d;

    t_xfer0 = wtime();
    checkCuda(cudaMemcpy(&d , deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost));
    t_transfer += wtime() - t_xfer0;

    delta = (float)d; // delta is now the sum of changed clusters


    delta /= numObjs;  // delta = sum of changed clusters / num of objects



    KERNEL_TIME_BEGIN();
    reduce_coord_clusters <<<numClusterBlocks, numThreadsPerClusterBlock, shared_reducer>>>
      (numCoords, numClusters, dev_sum_clusters, dev_sum_coords, final_sum_clusters, final_sum_coords, dev_num_reduce_clusters);
    KERNEL_TIME_END(acc_reduce_coord);
    checkLastCudaError();

    t_xfer0 = wtime();
    checkCuda(cudaMemcpy(cluster_count, final_sum_clusters, numClusters*sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(coord_sum, final_sum_coords, numCoords * numClusters * sizeof(float), cudaMemcpyDeviceToHost));
    t_transfer += wtime() - t_xfer0;

    for (int i=0; i<numClusters; i++) {
      for (int j=0; j<numCoords; j++) {
	int p = j + numCoords * i;
	coord_sum[p] = coord_sum[p]/cluster_count[i];

      }
    }

    for (int i=0; i<numCoords; i++) {
      for (int j=0; j<numClusters; j++) {
	int p = numCoords*j + i;
	dimClusters[i][j] = coord_sum[p];
//	printf("Centroids %d %0.2f\n", j, dimClusters[i][j]);
	
	}
    }

  //  printf("Delta %0.4f\n", delta);
   // printf("Looped %d\n", loop);
    
  } while ((delta > threshold) && (loop < 500));

  const double t_loop_sec = wtime() - t_loop_start;

  free(cluster_count);
  free(coord_sum);

#ifdef BENCH_KERNEL_BREAKDOWN
  checkCuda(cudaEventDestroy(k_start));
  checkCuda(cudaEventDestroy(k_stop));
#endif

  if (loop_iterations != NULL) *loop_iterations = loop;

  if (perf != NULL) {
    perf->iterations         = loop;
    perf->clustering_sec     = t_loop_sec;
    perf->transfer_sec       = t_transfer;
    perf->find_nearest_ms    = acc_find;
    perf->reduce_coord_ms    = acc_reduce_coord;
    perf->reduce_changed_ms  = acc_reduce_changed;
    perf->num_blocks         = (int) numClusterBlocks;
    perf->threads_per_block  = (int) numThreadsPerClusterBlock;
    perf->reduction_threads  = (int) numReductionThreads;
    perf->shared_bytes       = (size_t) clusterBlockSharedDataSize;
    perf->sm_count           = deviceProp.multiProcessorCount;
    perf->cc_major           = deviceProp.major;
    perf->cc_minor           = deviceProp.minor;
    snprintf(perf->gpu_name, sizeof(perf->gpu_name), "%s", deviceProp.name);
  }

  checkCuda(cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));

  checkCuda(cudaFree(deviceObjects));
  checkCuda(cudaFree(deviceClusters));
  checkCuda(cudaFree(deviceMembership));
  checkCuda(cudaFree(deviceIntermediates));
  checkCuda(cudaFree(dev_sum_coords));
  checkCuda(cudaFree(dev_sum_clusters));
  checkCuda(cudaFree(final_sum_clusters));
  checkCuda(cudaFree(final_sum_coords));
  checkCuda(cudaFree(dev_num_reduce_clusters));
    
  free(dimObjects[0]);
  free(dimObjects);

  // Convert clusters from [numCoords][numClusters] to [numClusters][numCoords]
  float **clusters;
  malloc2D(clusters, numClusters, numCoords, float);
  for (i = 0; i < numClusters; i++)
    for (j = 0; j < numCoords; j++)
      clusters[i][j] = dimClusters[j][i];
  *clusters_out = clusters;

  free(dimClusters[0]);
  free(dimClusters);

  cudaDeviceSynchronize();
    
  return 0;

}







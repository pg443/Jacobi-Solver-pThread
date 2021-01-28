/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 * Programmed by: Prasenjit Gaurav, Kidist Tessema
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "jacobi_solver.h"



/* Uncomment the line below to spit out debug information */ 
// #define DEBUG 
int hh = 0; 
int main(int argc, char **argv) 
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s matrix-size num_of_threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
		fprintf(stderr, "num_of_threads: number of parallel threads\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);
	int num_threads = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));
	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n");
	gettimeofday(&start, NULL);
	compute_using_pthreads(A, mt_solution_x, B, num_threads);
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
	gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));   
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads. 
 * Result must be placed in mt_sol_x. */
void compute_using_pthreads (const matrix_t A, matrix_t mt_sol_x, const matrix_t B, int num_threads)
{
    int max_iter = 100000; /* Maximum number of iterations to run */
	pthread_t *thread_id = (pthread_t *)malloc (num_threads * sizeof(pthread_t)); /* Data structure to store the thread IDs */
    pthread_attr_t attributes;      /* Thread attributes */
    pthread_attr_init(&attributes); /* Initialize thread attributes to default values */
    args_for_thread* thread_arg = (args_for_thread*)malloc(num_threads * sizeof(args_for_thread));
	double *ssd = (double*)malloc(num_threads * sizeof(double));
	BARRIER* barrier1 = (BARRIER*)malloc(sizeof(BARRIER));
	BARRIER* barrier2 = (BARRIER*)malloc(sizeof(BARRIER));


	barrier1->counter = 0;
    sem_init (&barrier1->counter_sem, 0, 1); /* Initialize the semaphore protecting the counter to 1 */
    sem_init (&barrier1->barrier_sem, 0, 0); /* Initialize the semaphore protecting the barrier to 0 */

    /* Initialize the barrier data structure */
    barrier2->counter = 0;
    sem_init (&barrier2->counter_sem, 0, 1); /* Initialize the semaphore protecting the counter to 1 */
    sem_init (&barrier2->barrier_sem, 0, 0); /* Initialize the semaphore protecting the barrier to 0 */

	int i;
	int num_elements = A.num_rows;
	int chunk = (int)floor((float)num_elements/(float)num_threads);

	for (i = 0; i < num_threads; i++){
		thread_arg[i].tid = i;
		thread_arg[i].A = A;
		thread_arg[i].B = B;
		thread_arg[i].x = mt_sol_x;
		thread_arg[i].barrier1 = barrier1;
		thread_arg[i].barrier2 = barrier2;
		thread_arg[i].partial_ssd = ssd;
		thread_arg[i].chunk = chunk;
		thread_arg[i].offset = i * chunk;
		thread_arg[i].iter = 0;
		thread_arg[i].num_of_threads = num_threads;
	}
	for (i = 0; i < num_threads; i++)
		pthread_create(&thread_id[i], &attributes, worker_thread, (void *)&thread_arg[i]);

	for (i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	int iter = thread_arg[num_threads-1].iter;
	if (iter < max_iter){
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", iter);
	}else{
        fprintf(stderr, "\nMaximum allowed iterations reached\n");
	}

	free((void *)ssd);
	free((void *)barrier1);
	free((void *)barrier2);
	free((void *)thread_id);
    free((void*)thread_arg);
}

void *worker_thread(void *args){
	args_for_thread *thread_data = (args_for_thread *)args;
	BARRIER* barrier1 = thread_data->barrier1;
	BARRIER* barrier2 = thread_data->barrier2;
	int max_iter = 100000; /* Maximum number of iterations to run */
	int i, j;
    int num_rows = thread_data->A.num_rows;
    int num_cols = thread_data->A.num_columns;

	/* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      

	int start = thread_data->offset;
	int end;
	if (thread_data->tid == thread_data->num_of_threads-1){
		end = num_rows;
	}else{
		end = thread_data->offset + thread_data->chunk;
	}

	/* Initialize current jacobi solution. */
    for (i = start; i < end; i++){
        thread_data->x.elements[i] = thread_data->B.elements[i];
	}
		
	sleep(1); 



	/* Perform Jacobi iteration. */
    int done = 0;
    double partial_ssd, ssd, mse;
    int num_iter = 0;
	while (!done) {
		for (i = start; i < end; i++) {
            double sum = 0.0;
            for (j = 0; j < num_cols; j++)
                if (i != j) sum += thread_data->A.elements[i * num_cols + j] * thread_data->x.elements[j];
            /* Update values for the unkowns for the current row. */
            new_x.elements[i] = (thread_data->B.elements[i] - sum)/thread_data->A.elements[i * num_cols + i];
        }

		/* Check for convergence and update the unknowns. */
		barrier_sync (barrier1, thread_data->tid, thread_data->num_of_threads);
        partial_ssd = 0.0; 
        for (i = start; i < end; i++) {
            partial_ssd += (new_x.elements[i] - thread_data->x.elements[i]) * (new_x.elements[i] - thread_data->x.elements[i]);
            thread_data->x.elements[i] = new_x.elements[i];
        }
		thread_data->partial_ssd[thread_data->tid] = partial_ssd;
		barrier_sync (barrier2, thread_data->tid, thread_data->num_of_threads);

		ssd = 0;
		for (i = 0; i < thread_data->num_of_threads; i++)
			ssd += thread_data->partial_ssd[i];

        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        
        if ((mse <= THRESHOLD) || (num_iter == max_iter)){
            done = 1;
			if (thread_data->tid == thread_data->num_of_threads-1) thread_data->iter = num_iter;
		}

	}
	free(new_x.elements);
	pthread_exit(NULL);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_rows + j]);
        }
        fprintf(stderr, "\n");
	} 
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}
/* The function that implements the barrier synchronization. */
void barrier_sync (BARRIER *barrier, int thread_number, int num_threads)
{
    sem_wait (&(barrier->counter_sem)); /* Obtain the lock on the counter */
	int i;
    /* Check if all threads before us, that is NUM_THREADS-1 threads have reached this point */
    if (barrier->counter == (num_threads - 1)) {
        barrier->counter = 0; /* Reset the counter */
					 
        sem_post (&(barrier->counter_sem)); 
					 
        /* Signal the blocked threads that it is now safe to cross the barrier */			 
        // printf("Thread number %d is signalling other threads to proceed. \n", thread_number); 			 
        for (i = 0; i < (num_threads - 1); i++)
            sem_post (&(barrier->barrier_sem));
    } 
    else {
        barrier->counter++; // Increment the counter
        sem_post (&(barrier->counter_sem)); // Release the lock on the counter
        sem_wait (&(barrier->barrier_sem)); // Block on the barrier semaphore and wait for someone to signal us when it is safe to cross
    }
}
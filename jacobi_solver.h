#ifndef _JACOBI_SOLVER_H_
#define _JACOBI_SOLVER_H_
#include <semaphore.h>
/* Programmed by: Prasenjit Gaurav, Kidist Tessema  */


#define THRESHOLD 1e-5      /* Threshold for convergence */
#define MIN_NUMBER 2        /* Min number in the A and b matrices */
#define MAX_NUMBER 10       /* Max number in the A and b matrices */

/* Matrix structure declaration */
typedef struct matrix_s {
    unsigned int num_columns;   /* Matrix width */
    unsigned int num_rows;      /* Matrix height */ 
    float *elements;
}  matrix_t;

typedef struct barrier_struct {
    sem_t counter_sem; /* Protects access to the counter */
    sem_t barrier_sem; /*Signals that barrier is safe to cross */
    int counter; /* The counter value */
} BARRIER;

/* Arguments struct for the threads */
typedef struct {
    matrix_t A;
    matrix_t B;
    matrix_t x;
    double* partial_ssd;
    BARRIER* barrier1;
    BARRIER* barrier2;
    int tid;
    int num_of_threads;
    int chunk;
    int offset;   
    int iter;
} args_for_thread;



/* Function prototypes */
matrix_t allocate_matrix (int, int, int);
extern void compute_gold(const matrix_t, matrix_t, const matrix_t, int);
extern void display_jacobi_solution(const matrix_t, const matrix_t, const matrix_t);
extern int iter();
int check_if_diagonal_dominant(const matrix_t);
matrix_t create_diagonally_dominant_matrix(int, int);
void compute_using_pthreads(const matrix_t, matrix_t, const matrix_t, int);
void print_matrix(const matrix_t);
float get_random_number(int, int);
void *worker_thread(void *);
void barrier_sync (BARRIER *, int, int);

#endif /* _JACOBI_SOLVER_H_ */


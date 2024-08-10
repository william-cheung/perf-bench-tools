/**
 *  This work is inspired by perf-bench. Author: Aimin Zhang
 *
 *  Reference:
 *    Cacheline Bouncing: https://www.quora.com/What-is-cache-line-bouncing-How-may-a-spinlock-trigger-this-frequently
 *    perf bench futex lock-pi (https://man7.org/linux/man-pages/man1/perf-bench.1.html)
 *    https://www.cs.rice.edu/~la5/doc/perf-doc/d1/d28/futex-lock-pi_8c_source.html
 * 
 *  Compilation:
 *    g++ -D_GNU_SOURCE -o bench_cache_contention bench_cacheline_contention.c -lpthread -lm -std=c++11 -DBENCH_INT
 * 
 */

#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <signal.h>
#include <errno.h>
#include <err.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <memory>

#ifdef BENCH_NOP
typedef int shared_var_type;
#define SHARED_VAR_INIT(var) (var) = 0
#define SHARED_VAR_OP(var, val)
#endif

#ifdef BENCH_INT
typedef int shared_var_type;
#define SHARED_VAR_INIT(var) (var) = 0
#define SHARED_VAR_OP(var, val) (var) = (val)
#endif

#ifdef BENCH_ATOMIC_INT
typedef std::atomic_int shared_var_type;
#define SHARED_VAR_INIT(var) (var) = 0
#define SHARED_VAR_OP(var, val) (var).fetch_add(val)
#endif

#ifdef BENCH_SHARED_PTR
typedef std::shared_ptr<int> shared_var_type;
#define SHARED_VAR_INIT(var) (var) = std::make_shared<int>(0);
#define SHARED_VAR_OP(var, val) shared_var_type temp = (var)
#endif


#ifndef NTHREADS
#define NTHREADS 0
#endif


struct stats {
    double n, mean, M2;
    uint64_t max, min;
};


struct worker {
    int tid;
    shared_var_type *var;
    pthread_t thread;
    unsigned long ops;
};

static shared_var_type global_shared_var;
static struct worker *worker;
static unsigned int nsecs = 10;
static int done = 0;
static unsigned int nthreads = NTHREADS;
static unsigned int ncpus = 0;
struct timeval start, end, runtime;
static pthread_mutex_t thread_lock;
static unsigned int threads_starting;
static struct stats throughput_stats;
static pthread_cond_t thread_parent, thread_worker;


static inline void init_stats(struct stats *stats)
{
    stats->n    = 0.0;
    stats->mean = 0.0;
    stats->M2   = 0.0;
    stats->min  = (uint64_t) -1;
    stats->max  = 0;
}


void update_stats(struct stats *stats, uint64_t val)
{
    double delta;

    stats->n++;
    delta = val - stats->mean;
    stats->mean += delta / stats->n;
    stats->M2 += delta*(val - stats->mean);

    if (val > stats->max)
        stats->max = val;

    if (val < stats->min)
        stats->min = val;
}

double avg_stats(struct stats *stats)
{
    return stats->mean;
}

/*
 * http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 *
 *       (\Sum n_i^2) - ((\Sum n_i)^2)/n
 * s^2 = -------------------------------
 *                  n - 1
 *
 * http://en.wikipedia.org/wiki/Stddev
 *
 * The std dev of the mean is related to the std dev by:
 *
 *             s
 * s_mean = -------
 *          sqrt(n)
 *
 */
double stddev_stats(struct stats *stats)
{
    double variance, variance_mean;

    if (stats->n < 2)
        return 0.0;

    variance = stats->M2 / (stats->n - 1);
    variance_mean = variance / stats->n;

    return sqrt(variance_mean);
}


double rel_stddev_stats(double stddev, double avg)
{
    double pct = 0.0;
 
    if (avg)
        pct = 100.0 * stddev/avg;
 
    return pct;
}

static void print_summary(void)
{
    unsigned long avg = avg_stats(&throughput_stats);
    double stddev = stddev_stats(&throughput_stats);

    printf("\nAveraged %ld operations/sec (+- %.2f%%), total secs = %d\n",
           avg, rel_stddev_stats(stddev, avg),
           (int) runtime.tv_sec);
}


static void toggle_done(int sig,
            siginfo_t *info,
            void *uc)
{
    /* inform all threads that we're done for the day */
    done = 1;
    gettimeofday(&end, NULL);
    timersub(&end, &start, &runtime);
}

static void *workerfn(void *arg)
{
    struct worker *w = (struct worker *) arg;
    unsigned long ops = w->ops;

    pthread_mutex_lock(&thread_lock);
    threads_starting--;
    if (!threads_starting)
        pthread_cond_signal(&thread_parent);
    pthread_cond_wait(&thread_worker, &thread_lock);
    pthread_mutex_unlock(&thread_lock);

    do {
        SHARED_VAR_OP(*(worker->var), worker->tid);
        ops++; /* account for thread's share of work */
    }  while (!done);

    w->ops = ops;
    return NULL;
}

static void create_threads(struct worker *w, pthread_attr_t thread_attr)
{
    cpu_set_t cpuset;
    unsigned int i;

    threads_starting = nthreads;

    for (i = 0; i < nthreads; i++) {
        worker[i].tid = i;

   
        worker[i].var = &global_shared_var;

        CPU_ZERO(&cpuset);
        CPU_SET(i % ncpus, &cpuset);

        if (pthread_attr_setaffinity_np(&thread_attr, sizeof(cpu_set_t), &cpuset))
            err(EXIT_FAILURE, "pthread_attr_setaffinity_np");

        if (pthread_create(&w[i].thread, &thread_attr, workerfn, &worker[i]))
            err(EXIT_FAILURE, "pthread_create");
    }
}

int main(int argc, const char **argv)
{
    int ret = 0;
    unsigned int i;
    struct sigaction act;
    pthread_attr_t thread_attr;

    SHARED_VAR_INIT(global_shared_var);    

    ncpus = get_nprocs();
    if (!nthreads)
        nthreads = ncpus;  

    sigfillset(&act.sa_mask);
    act.sa_sigaction = toggle_done;
    sigaction(SIGINT, &act, NULL);

    worker = (struct worker*) calloc(nthreads, sizeof(*worker));
    if (!worker)
        err(EXIT_FAILURE, "calloc");

    printf("Run summary [PID %d]: %d threads doing cacheline bouncing test for %d secs.\n\n",
           getpid(), nthreads, nsecs);

    init_stats(&throughput_stats);
    pthread_mutex_init(&thread_lock, NULL);
    pthread_cond_init(&thread_parent, NULL);
    pthread_cond_init(&thread_worker, NULL);

    threads_starting = nthreads;
    pthread_attr_init(&thread_attr);
    gettimeofday(&start, NULL);

    create_threads(worker, thread_attr);
    pthread_attr_destroy(&thread_attr);

    pthread_mutex_lock(&thread_lock);
    while (threads_starting)
        pthread_cond_wait(&thread_parent, &thread_lock);
    pthread_cond_broadcast(&thread_worker);
    pthread_mutex_unlock(&thread_lock);

    sleep(nsecs);
    toggle_done(0, NULL, NULL);

    for (i = 0; i < nthreads; i++) {
        ret = pthread_join(worker[i].thread, NULL);
        if (ret)
            err(EXIT_FAILURE, "pthread_join");
    }

    /* cleanup & report results */
    pthread_cond_destroy(&thread_parent);
    pthread_cond_destroy(&thread_worker);
    pthread_mutex_destroy(&thread_lock);

    for (i = 0; i < nthreads; i++) {
        unsigned long t = worker[i].ops/runtime.tv_sec;

        update_stats(&throughput_stats, t);
        printf("[thread %3d] %ld ops/sec\n",
               worker[i].tid, t);

    }

    print_summary();

    free(worker);
    return ret;
}

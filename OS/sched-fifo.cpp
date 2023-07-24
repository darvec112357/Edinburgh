/*
 * FIFO Scheduling Algorithm
 * SKELETON IMPLEMENTATION -- TO BE FILLED IN FOR TASK (2)
 */

/*
 * STUDENT NUMBER: s1810054
 */
#include <infos/kernel/sched.h>
#include <infos/kernel/thread.h>
#include <infos/kernel/log.h>
#include <infos/util/list.h>
#include <infos/util/lock.h>

using namespace infos::kernel;
using namespace infos::util;

/**
 * A FIFO scheduling algorithm
 */
class FIFOScheduler : public SchedulingAlgorithm
{
public:
	/**
	 * Returns the friendly name of the algorithm, for debugging and selection purposes.
	 */
	const char* name() const override { return "fifo"; }

	/**
	 * Called when a scheduling entity becomes eligible for running.
	 * @param entity
	 */
	void add_to_runqueue(SchedulingEntity& entity) override
	{
		UniqueIRQLock l;
		runqueue.enqueue(&entity);
	}

	/**
	 * Called when a scheduling entity is no longer eligible for running.
	 * @param entity
	 */
	void remove_from_runqueue(SchedulingEntity& entity) override
	{
		UniqueIRQLock l;
		runqueue.remove(&entity);
	}

	/**
	 * Called every time a scheduling event occurs, to cause the next eligible entity
	 * to be chosen.  The next eligible entity might actually be the same entity, if
	 * e.g. its timeslice has not expired, or the algorithm determines it's not time to change.
	 */
	SchedulingEntity *pick_next_entity() override
	{
		//In fifo algorithm, only the first entity will be run until completion,
		//after which it will be removed.
		if (runqueue.count() == 0) return NULL;
		else return runqueue.first();
	}

	//Bonus question:
	//When the main thread is generated, it will be added to the back of the runqueue by fifo algorithm.
	//The only way to stop thread 1 is for the main thread to receive the <Enter> command.
	//However, during the test, thread 1 will always be occupying the CPU. Main thread will never be run,
	//which means that <Enter> will never be read. Thus, thread 1 will never be stopped.

private:
	// A list containing the current runqueue.
	List<SchedulingEntity *> runqueue;
};

/* --- DO NOT CHANGE ANYTHING BELOW THIS LINE --- */

RegisterScheduler(FIFOScheduler);

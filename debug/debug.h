#ifndef _DEBUG_H_
#define _DEBUG_H_

#include <stdio.h>

typedef enum {
	ERROR = 0,
	WARN  = 1,
	INFO  = 2,
	DEBUG = 3,
}LOG_LEVEL;

/* function prototypes */
void log_init(LOG_LEVEL level);
void log_term(void);
void debug_msg(const char *fmt, ...);
void log_msg(LOG_LEVEL level, const char *fmt, ...);
void set_log_level(LOG_LEVEL level);

#endif

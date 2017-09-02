#ifndef _DEBUG_H_
#define _DEBUG_H_

#include <stdio.h>

typedef enum {
	INFO,
	WARN,
	ERROR
}LOG_LEVEL;

/* function prototypes */
void log_init(void);
void log_term(void);
void debug_msg(const char *fmt, ...);
void log_msg(LOG_LEVEL level, const char *fmt, ...);


#endif

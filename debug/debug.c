#include <stdlib.h>
#include <stdarg.h>
#include <time.h>

#include "debug.h"

FILE *LOG_FP;

signed char DEBUG_ON = 0;

const char *LOG_FILE = "./raytracer.log";

void log_init(void)
{
	time_t now = 0;
	char *date = NULL;
	LOG_FP = NULL;
	LOG_FP = fopen(LOG_FILE, "a");
	if(LOG_FP == NULL) {
		debug_msg("debug messages cannot be written to a file.\n");
		return;
	}
	DEBUG_ON = 1;
	now = time(NULL);
	date = ctime(&now);
	log_msg(INFO, "Opened Log file: %s at local system time: %s\n", LOG_FILE, date);
}

void log_term(void)
{
	fclose(LOG_FP);
}

void debug_msg(const char *fmt, ...)
{
	if(DEBUG_ON) {
		if(fmt != NULL) {
			va_list args;
			va_start(args, fmt);
			vfprintf(stderr, fmt, args);
			va_end(args);
		}
	}
}

void log_msg(LOG_LEVEL level, const char *fmt, ...)
{
	if(DEBUG_ON) {
		if(fmt != NULL) {
			va_list args;
			va_start(args, fmt);
			char *mod_fmt = malloc((sizeof(char) * strlen(fmt)) + 6);
			switch(level) {
 			case INFO:
				strcat(mod_fmt, );
				vfprintf(stdout, mod_fmt, args);
				break;
			case WARN:
				vfprintf(stdout, mod_fmt, args);
				break;
			case ERROR:
				vfprintf(stderr, mod_fmt, args);
				break;
			case default:
				vfprintf(stdout, mod_fmt, args);
				break;
			}
			if(LOG_FP != NULL) {
				vfprintf(LOG_FP, mod_fmt, args);
			}
			va_end(args);
		}
	}
}

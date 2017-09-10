#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>

#include "debug.h"

FILE *LOG_FP;
time_t LOG_NOW;
char *LOG_DATE;

signed char DEBUG_ON = 0;

const char *LOG_FILE = "./raytracer.log";

void log_init(void)
{
	LOG_NOW = 0;
	LOG_DATE = NULL;
	LOG_FP = NULL;
	LOG_FP = fopen(LOG_FILE, "a");
	if(LOG_FP == NULL) {
		debug_msg("debug messages cannot be written to a file.\n");
		return;
	}
	DEBUG_ON = 1;
	LOG_NOW = time(NULL);
	LOG_DATE = ctime(&LOG_NOW);
	log_msg(INFO, "Opened Log file: %s", LOG_FILE);
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
			LOG_NOW = time(NULL);
			LOG_DATE = strtok(ctime(&LOG_NOW), "\n");
			char *mod_fmt = malloc((sizeof(char) * strlen(fmt)) +
					       (sizeof(char) * 12) +
					       (sizeof(char) * strlen(LOG_DATE)));
			switch(level) {
 			case INFO:
				strcpy(mod_fmt, LOG_DATE);
				strcat(mod_fmt, " - INFO: ");
				strcat(mod_fmt, fmt);
				vfprintf(stdout, mod_fmt, args);
				break;
			case WARN:
				strcpy(mod_fmt, LOG_DATE);
				strcat(mod_fmt, " - WARN: ");
				strcat(mod_fmt, fmt);
				vfprintf(stdout, mod_fmt, args);
				break;
			case ERROR:
				strcpy(mod_fmt, LOG_DATE);
				strcat(mod_fmt, " - ERROR: ");
				strcat(mod_fmt, fmt);
				vfprintf(stderr, mod_fmt, args);
				break;
			default:
				strcpy(mod_fmt, LOG_DATE);
				strcat(mod_fmt, " - INFO: ");
				strcat(mod_fmt, fmt);
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

/*
 * main.cc
 *
 *  Created on: Nov 9, 2025
*/

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"

/* Command definitions for command line options */
#define TRAIN 1
#define PREDICT 2

/* parseCmd: parse command line arguments
 * Arguments:
 *   argc: argument count
 *   argv: argument vector
 * Returns:
 *   TRAIN or PREDICT based on user input, 0 for invalid input
 */
int parseCmd(int argc,char** argv){
	if (argc!=2) return 0;
	if (strcmp(argv[1],"train")==0)return TRAIN;
	if (strcmp(argv[1],"predict")==0) return PREDICT;
	return 0;
}

/* usage: print usage information
 * Returns:
 *   0
 */
int usage(){
	printf("Usage: nnp [train|predict]\n\tNote: predict requires a previously trained model in the directory named model.bin\n");
	return 0;
}

/* train: load dataset, train model, save model
 * Returns:
 *   void	
*/
void train(){
	load_dataset();
	MODEL model;
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	train_model(&model);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
	save_model(&model);
	printf("Trained model in %.3f seconds\n", elapsed);
}
/* predict_test: load dataset, load model, predict on test set
 * Returns:
 *   void	
*/
void predict_test(){
	load_dataset();
	MODEL model;
	load_model(&model);
	for (int i=0;i<NUM_TEST;i++){
		predict(test_data[i],&model); 
	}
}

/* main: entry point of the program
 * Arguments:
 *   argc: argument count
 *   argv: argument vector
 * Returns:
 *   0 on success, usage information on invalid input
*/
int main(int argc,char** argv){
	switch (parseCmd(argc,argv)){
		case TRAIN:{
			train();
	      		break;
		}
		case PREDICT:{
			predict_test();
			break;
		}
		default:{
		        return usage();
		}
	}
}

/******************************************************************************

Solving TSP using Hopfield Network
ECE 559 (Neural Networks)
PADMAGANDHA SAHOO
11th Dec ’03

Dr Antti J Ylikoski 2020-09-11

From the Daniel Graupe book, 2nd edition


******************************************************************************/

// TSP.H


#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <fstream>

#define Maxsize 30


class neuron
{
 protected:
  int cit,ord;
  float output;
  float activation;
  friend class HP_network;
 public:
  neuron() {};
  void getnrn(int,int);
};





class HP_network
{
 public:
  int cityno;
  //Number of City
  float a,b,c,d,totout,distnce;
  neuron (tnrn)[Maxsize][Maxsize];
  int dist[Maxsize][Maxsize];
  int tourcity[Maxsize];
  int tourorder[Maxsize];
  float outs[Maxsize][Maxsize];
  float acts[Maxsize][Maxsize];
  float weight[Maxsize*Maxsize][Maxsize*Maxsize];
  float citouts[Maxsize];
  float ordouts[Maxsize];
  float energy;
  HP_network() { };
  void getnwk(int,float,float,float,float);
  void initdist(int);
  void findtour();
  void asgninpt(float *);
  void calcdist();
  void iterate(int,int,float,float,float);
  void getacts(int,float,float);
  void getouts(float);
  float getenergy();
  void print_dist();
 void print_weight(int);
 void print_tour();
 void print_acts();
 void print_outs();
};


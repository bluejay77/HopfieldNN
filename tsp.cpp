//TSP.CPP

/*

From the Daniel Graupe book, 2nd edition

The Traveling Salesman Problem in the C++ language

Dr Antti J Ylikoski 2020-09-11

Use:

g++ tsp.cpp -o tsp
./tsp

This one needs debugging and adaptation.

*/



#include "tsp.h"
#include <stdlib.h>
#include <time.h>



int randomnum(int maxval) // Create random numbers between 1 to 100
{
  return rand()%maxval;
}


/* ========= Compute the Kronecker delta function ======================= */
int krondelt(int i,int j)
{
  int k;
  k=((i==j)?(1):(0));
  return k;
}



/* ========== compute the distance between two co-ordinates =============== */
int distance(int x1,int x2,int y1,int y2)
{
  int x,y,d;
  x=x1-x2;
  x=x*x;
  y=y1-y2;
  y=y*y;
  d=(int)sqrt(x+y);
  return d;
}



void neuron::getnrn(int i,int j)
{
  cit=i;
  ord=j;
  output=0.0;
  activation=0.0;
}



/* ====== Randomly generate the co-ordinates of the cities =============== */


 //initiate the distances between the k cities

void HP_network::initdist(int cityno)
{
  int i,j;
  int rows=cityno, cols=2;
  int **ordinate;
  int **row;
  ordinate = (int **)malloc((rows+1) *sizeof(int *)); /*one extra for sentinel*/
  /* now allocate the actual rows */

  for(i = 0; i < rows; i++)
    {
      ordinate[i] = (int *)malloc(cols * sizeof(int));
    }
  /* initialize the sentinel value */
  ordinate[rows] = 0;
  srand(cityno);
  for(i=0; i<rows; i++)
    {
      ordinate[i][0] = rand() % 100;
      ordinate[i][1] = rand() % 100;
    }
  std::cout <<"\nThe Co-ordinates of "<<cityno<<" cities: \n";
  for (i=0;i<cityno;++i)
    {
      std::cout <<"X "<<i<<": "<<ordinate[i][0]<<" ";
      std::cout <<"Y "<<i<<": "<<ordinate[i][1]<<"\n";
    }
  for (i=0;i<cityno;++i)
    {
      dist[i][i]=0;
      for (j=i+1;j<cityno;++j)
	{
	  dist[i][j]=distance(ordinate[i][0],ordinate[j][0],
			      ordinate[i][1],ordinate[j][1])/1;
	}
    }
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<i;++j)
	{
	  dist[i][j]=dist[j][i];
	}
    }
  print_dist();
  std::cout << "\n";
  //print the distance matrix
  for(row = ordinate; *row != 0; row++)
    {
      free(*row);
    }
  free(ordinate);
}


/* ============== Print Distance Matrix ==================== */


void HP_network::print_dist()
{
  int i,j;
  std::cout <<"\n Distance Matrix\n";
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<cityno;++j)
	{
	  std::cout << dist[i][j] <<" ";
	}
      std::cout << "\n";
    }
}


/* ============ Compute the weight matrix ==================== */

void HP_network::getnwk(int citynum,float x,float y,float z,float w)
{
  int i,j,k,l,t1,t2,t3,t4,t5,t6;
  int p,q;
  cityno=citynum;
  a=x;
  b=y;
  c=z;
  d=w;
  initdist(cityno);
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<cityno;++j)
	{
	  tnrn[i][j].getnrn(i,j);
	}
    }
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<cityno;++j)
	{
	  p=((j==cityno-1)?(0):(j+1));
	  q=((j==0)?(cityno-1):(j-1));
	  t1=j+i*cityno;
	  for (k=0;k<cityno;++k)
	    {
	      for (l=0;l<cityno;++l)
		{
		  t2=l+k*cityno;
		  t3=krondelt(i,k);
		  t4=krondelt(j,l);
		  t5=krondelt(l,p);
		  t6=krondelt(l,q);
		  weight[t1][t2]=-a*t3*(1-t4)-b*t4*(1-t3)
		    -c-d*dist[i][k]*(t5+t6)/100;
		}
	    }
	}
    }
}



void HP_network::print_weight(int k)
{
  int i,j,nbrsq;
  nbrsq=k*k;
  std::cout <<" \nWeight Matrix\n";
  std::cout << "\nWeight Matrix\n";
  for (i=0;i<nbrsq;++i)
    {
      for (j=0;j<nbrsq;++j)
	{
	  std::cout << weight[i][j] << " ";
	}
      std::cout <<" \n";
    }
}


/* =========== Assign initial inputs to the network ============= */

void HP_network::asgninpt(float *ip)
{
  int i,j,k,l,t1,t2;
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<cityno;++j)
	{
	  acts[i][j]=0.0;
	}
    }
  //find initial activations
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<cityno;++j)
	{
	  t1=j+i*cityno;
	  for (k=0;k<cityno;++k)
	    {
	      for (l=0;l<cityno;++l)
		{
		  t2=l+k*cityno;
		  acts[i][j]+=weight[t1][t2]*ip[t1];
		}
	    }
	}
    }
}



/* ======== Compute the activation function outputs =================== */

void HP_network::getacts(int nprm,float dlt,float tau)
{
  int i,j,k,p,q;
  float r1,r2,r3,r4,r5;
  r3=totout-nprm;
  for (i=0;i<cityno;++i)
    {
      r4=0.0;
      p=((i==cityno-1)?(0):(i+1));
      q=((i==0)?(cityno-1):(i-1));
      for (j=0;j<cityno;++j)
	{
	  r1=citouts[i]-outs[i][j];
	  r2=ordouts[j]-outs[i][j];
	  for (k=0;k<cityno;++k)
	    {
	      r4+=dist[i][k]*(outs[k][p]+outs[k][q])/100;
	    }
	  r5=dlt*(-acts[i][j]/tau-a*r1-b*r2-c*r3-d*r4);
	  acts[i][j]+=r5;
	}
    }
}

/* ============== Get Neural Network Output ===================== */

void HP_network::getouts(float la)
{
  double b1,b2,b3,b4;
  int i,j;
  totout=0.0;
  for (i=0;i<cityno;++i)
    {
      citouts[i]=0.0;
      for (j=0;j<cityno;++j)
	{
	  b1=la*acts[i][j];
	  b4=b1;
	  b2=exp(b4);
	  b3=exp(-b4);
	  outs[i][j]= (float)(1.0+(b2-b3)/(b2+b3))/2.0;
	  citouts[i]+=outs[i][j];
	}
      totout+=citouts[i];
    }
  for (j=0;j<cityno;++j)
    {
      ordouts[j]=0.0;
      for (i=0;i<cityno;++i)
	{
	  ordouts[j]+=outs[i][j];
	}
    }
}


/* ============ Compute the Energy function ======================= */

float HP_network::getenergy()
{
  int i,j,k,p,q;
  float t1,t2,t3,t4,e;
  t1=0.0;
  t2=0.0;
  t3=0.0;
  t4=0.0;
  for (i=0;i<cityno;++i)
    {
      p=((i==cityno-1)?(0):(i+1));
      q=((i==0)?(cityno-1):(i-1));
      for (j=0;j<cityno;++j)
	{
	  t3+=outs[i][j];
	  for (k=0;k<cityno;++k)
	    {
	      if (k!=j)
		{
		  t1+=outs[i][j]*outs[i][k];
		  t2+=outs[j][i]*outs[k][i];
		  t4+=dist[k][j]*outs[k][i]
		    *(outs[j][p]+outs[j][q])/10;
		}
	    }
	}
    }
  t3=t3-cityno;
  t3=t3*t3;
  e=0.5*(a*t1+b*t2+c*t3+d*t4);
  return e;
}

/* ======== find a valid tour ========================= */

void HP_network::findtour()
{
  int i,j,k,tag[Maxsize][Maxsize];
  float tmp;
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<cityno;++j)
	{
	  tag[i][j]=0;
	}
    }
  for (i=0;i<cityno;++i)
    {
      tmp=-10.0;
      for (j=0;j<cityno;++j)
	{
	  for (k=0;k<cityno;++k)
	    {
	      if ((outs[i][k]>=tmp)&&(tag[i][k]==0))
		tmp=outs[i][k];
	    }
	  if ((outs[i][j]==tmp)&&(tag[i][j]==0))
	    {
	      tourcity[i]=j;
	      tourorder[j]=i;
	      std::cout << "tour order" << j << "\n";
	      for (k=0;k<cityno;++k)
		{
		  tag[i][k]=1;
		  tag[k][j]=1;
		}
	    }
	}
    }
}



//print outputs
void HP_network::print_outs()
{
  int i,j;
  std::cout << "\n the outputs\n";
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<cityno;++j)
	{
	  std::cout << outs[i][j] << " ";
	}
      std::cout << "\n";
    }
}

/* ======= Calculate total distance for tour ============== */
void HP_network::calcdist()
{
  int i,k,l;
  distnce=0.0;
  for (i=0;i<cityno;++i)
    {
      k=tourorder[i];
      l=((i==cityno-1)?(tourorder[0]):(tourorder[i+1]));
      distnce+=dist[k][l];
    }
  std::cout << "\nTotal distance of tour is : "<< distnce << "\n";
}


/* ======= Print Tour Matrix ============================== */
void HP_network::print_tour()
{
  int i;
  std::cout << "\nThe tour order: \n";
  for (i=0;i<cityno;++i)
    {
      std::cout << tourorder[i] << " ";
      std::cout << "\n";
    }
}


/* ======= Print network activations ======================== */
void HP_network::print_acts()
{
  int i,j;
  std::cout << "\n the activations:\n";
  for (i=0;i<cityno;++i)
    {
      for (j=0;j<cityno;++j)
	{
	  std::cout <<acts[i][j]<<" ";
	}
      std::cout <<"\n";
    }
}


/*========== Iterate the network specified number of times =============== */
void HP_network::iterate(int nit,int nprm,float dlt,float tau,float la)
{
  int k,b;
  double oldenergy,newenergy, energy_diff;
  b=1;
  oldenergy=getenergy();
  std::cout <<""<<oldenergy<<"\n";
  k=0;
  do
    {
      getacts(nprm,dlt,tau);
      getouts(la);
      newenergy=getenergy();
      std::cout <<""<<newenergy<<"\n";
      //energy_diff = oldenergy - newenergy;
      //if (energy_diff < 0)
      //
      energy_diff = energy_diff*(-1);
      if (oldenergy - newenergy < 0.0000001)
	{
	  //printf("\nbefore break: %lf\n", oldenergy - newenergy);
	  break;
	}
      oldenergy = newenergy;
      k++;
    }
  while (k<nit) ;
  std::cout <<"\n"<<k<<" iterations taken for convergence\n";
  //print_acts();
  //outFile<<"\n";
  //print_outs();
  //outFile<<"\n";
}


void hidden_main()
{
/*===== Constants used in Energy, Weight and Activation Matrix =========== */
  int nprm=15;
  float a=0.5;
  float b=0.5;
  float c=0.2;
  float d=0.5;
  double dt=0.01;
  float tau=1;
  float lambda=3.0;
  int i,n2;
  int numit=4000;
  int cityno=15;
  //
  std::cin>>cityno; //No. of cities
  float input_vector[Maxsize*Maxsize];
  time_t start,end;
  double dif;
  start = time(NULL);
  srand((unsigned)time(NULL));
  //time (&start);
  n2=cityno*cityno;
  std::cout<<"Input vector:\n";
  for (i=0;i<n2;++i)
    {
      if (i%cityno==0)
	{
	  std::cout<<"\n";
	}
      input_vector[i]=(float)(randomnum(100)/100.0)-1;
      std::cout<<input_vector[i]<<" ";
    }
  std::cout <<"\n";
  //create HP_network and operate
  HP_network *TSP_NW=new HP_network;
  if (TSP_NW==0)
    {
      std::cout<<"not enough memory\n";
      exit(1);
    }
  TSP_NW->getnwk(cityno,a,b,c,d);
  TSP_NW->asgninpt(input_vector);
  TSP_NW->getouts(lambda);
  //TSP_NW->print_outs();
  TSP_NW->iterate(numit,nprm,dt,tau,lambda);
  TSP_NW->findtour();
  TSP_NW->print_tour();
  TSP_NW->calcdist();
  //time (&end);
  end = time(NULL);
  dif = end - start;
  printf("Time taken to run this simulation: %lf\n",dif);
}


// The dummy main()

int main()
{
  printf("Hello, world!\n!");
}


#include "OptimizerGD.h"
#include "Objective.h"
#include <iostream>

using namespace std;
using namespace Eigen;

OptimizerGD::OptimizerGD()
{
	
}

OptimizerGD::~OptimizerGD()
{
	
}

VectorXd OptimizerGD::optimize(const shared_ptr<Objective> objective, const VectorXd &xInit)
{
    
    VectorXd newX=xInit;
    int n=(int)xInit.size();
    VectorXd dx;
    
    for(iter=0;iter<iterMax;iter++){
        
        VectorXd g(n);
        double f = objective->evalObjective(newX, g);
        VectorXd p = -g;
        if(abs(alphaInit-1.0)<tol){
            // line search
            double a=alphaInit;
            for(int iterLS=0;iterLS<iterMaxLS;iterLS++){
                dx = a*p;
                double f1=objective->evalObjective(newX+dx);
                if(f1<f){
                    break;
                }
                a*=gamma;
            }
        }
        else{
            dx=alphaInit*p;
        }
        newX=newX+dx;
        if(g.norm()<tol){
            break;
        }
    }
	return newX;
}

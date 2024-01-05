#include "OptimizerNM.h"
#include "Objective.h"
#include <iostream>

using namespace std;
using namespace Eigen;

OptimizerNM::OptimizerNM()
{
	
}

OptimizerNM::~OptimizerNM()
{
	
}

VectorXd OptimizerNM::optimize(const shared_ptr<Objective> objective, const VectorXd &xInit)
{
    VectorXd newX=xInit;
    int n=(int)xInit.size();
    VectorXd dx;
    VectorXd ans;
    
    
    for(iter=0;iter<iterMax;iter++){
        
        VectorXd g(n);
        MatrixXd H(n,n);
        double f = objective->evalObjective(newX, g, H);
        VectorXd p = (-H).ldlt().solve(g);
        double alpha=alphaInit;
        for(int iterLS=0;iterLS<iterMaxLS;iterLS++){
            dx=alpha*p;
            double f1=objective->evalObjective(newX+dx);
            if(f1<f)break;
            alpha*=gamma;
        }
        newX+=dx;
        if(g.norm()<tol)break;
    }
    
	return newX;
}

#include "ObjectiveRosenbrock.h"
#include <cmath>

using namespace Eigen;

ObjectiveRosenbrock::ObjectiveRosenbrock(double a, double b)
{
	this->a = a;
	this->b = b;
}

ObjectiveRosenbrock::~ObjectiveRosenbrock()
{
	
}

double ObjectiveRosenbrock::evalObjective(const VectorXd &x) const
{
    double x0=x(0),x1=x(1);
    double f=pow((a-x0),2)+b*pow(x1-pow(x0,2),2);
    return f;
}

double ObjectiveRosenbrock::evalObjective(const VectorXd &x, VectorXd &g) const
{
    double x0=x(0),x1=x(1);
    double f=pow((a-x0),2)+b*pow(x1-pow(x0,2),2);
    g<<-2*(a-x0)-4*b*x0*(x1-pow(x0,2)),2*b*(x1-pow(x0,2));
    return f;
}

double ObjectiveRosenbrock::evalObjective(const VectorXd &x, VectorXd &g, MatrixXd &H) const
{
    double x0=x(0),x1=x(1);
    double f=pow((a-x0),2)+b*pow(x1-pow(x0,2),2);
    g<<-2*(a-x0)-4*b*x0*(x1-pow(x0,2)),2*b*(x1-pow(x0,2));
    H<<2*(2*b*(3*pow(x0,2)-x1)+1),2*-2*b*x0,
    2*-2*b*x0, 2*b;
	return f;
}


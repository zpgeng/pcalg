\name{pcAlgo}
\alias{pcAlgo}
\title{PC-Algorithm: Estimate the Underlying Graph (Skeleton) or
  Equivalence Class (CPDAG) of a DAG}
\description{
  Estimate the underlying graph (\code{\link[graph]{ugraph}} or
  \dQuote{skeleton} (structure) of a DAG (\bold{D}irected \bold{A}cyclic
  \bold{G}raph) from data using the PC-algorithm. Alternatively, one can
  directly infer the equivalence class of the underlying structure,
  which is uniquely represented by its CPDAG (\bold{C}ompleted
  \bold{P}artially \bold{D}irected \bold{A}cyclic \bold{G}raph).
}
\usage{
pcAlgo(dm, alpha, corMethod = "standard", verbose = FALSE, directed = FALSE)
}
\arguments{
  \item{dm}{data matrix; rows correspond to samples, cols correspond to nodes.}
  \item{alpha}{significance level for the individual partial correlation tests.}
  \item{corMethod}{a character string speciyfing the method for
    (partial) correlation estimation.
    "standard", "Qn" or "ogkQn" for standard and robust (based on
    the Qn scale estimator without and with OGK) correlation estimation}
  \item{verbose}{Indicates whether some intermediate output should be
    shown (WARNING: This decreases the performance dramatically!)}
  \item{directed}{If \code{FALSE}, the underlying skeleton is computed;
    if \code{TRUE}, the underlying CPDAG is computed}
}
\value{
  An object of \code{\link{class}} \code{"pcAlgo"} (see
  \code{\link[pcAlgo]{pcAlog-class}}) containing an undirected graph
  (object of \code{\link{class}} \code{"graph"}, see 
  \code{\link[graph]{graph-class}} from the package \pkg{graph})
  (without weigths) as estimate of the skeleton or the CPDAG of the
  underlying DAG. 
}
\details{
  The algorithm starts with a complete undirected graph. In a first
  sweep, an edge ij is kept only if \eqn{H_0: Cor(X_i,X_j) = 0} can be
  rejected on significance level \code{alpha}.  All ordered pairs ij of
  nodes  of the
  resulting graph are then swept again.  An edge ij is kept only if
  \eqn{H_0: Cor(X_i,X_j|X_k)=0} can be rejected for all neighbours k of
  i in the current graph. Again, the remaining egdes are swept.  This
  time, an ordered pair (edge) ij is 
  kept only if \eqn{H_0: Cor(X_i,X_j|X_a,X_b)=0} can be rejected for all
  subsets of size two \eqn{(a,b)} of the neighbours of i in the
  remaining graph.  In the next
  step, the remaining edges are tested using all subsets of size three,
  then of size four and so on.  The algorithm stops when the largest
  neighbourhood is smaller than the size of the conditioning sets.

  The partial correlations are 
  computed recursively or via matrix inversion from the correlation matrix,
  which are computed by the 
  specified method (\code{corMethod}).  The partial correlation tests
  are based on Fisher's z-transformation.  For more details on the
  methods for computing the correlations see \code{\link{mcor}}.
}
\references{
  P. Spirtes, C. Glymour and R. Scheines (2000)
  \emph{Causation, Prediction, and Search}, 2nd edition, The MIT Press.

  Kalisch M. and P. B\"uhlmann (2007)
  \emph{Estimating high-dimensional
    directed acyclic graphs with the PC-algorithm};
    JMLR, Vol. 8, 613-636, 2007.
}
\seealso{\code{\link{randomDAG}} for generating a random DAG;
  \code{\link{rmvDAG}} for generating data according to a DAG;
  \code{\link{compareGraphs}} for comparing undirected graphs in terms of
  TPR, FPR and TDR.  Further, \code{\link[graph]{randomGraph}} (in
  package \pkg{graph}) for other random graph
  models. \code{\link{udag2cpdag}} for converting the skeleton to a CPDAG.
}
\author{
  Markus Kalisch (\email{kalisch@stat.math.ethz.ch}) and Martin Maechler.
}
\examples{
p <- 10
## generate and draw random DAG :
set.seed(101)
class(myDAG <- randomDAG(p, prob = 0.2))
plot(myDAG, main = "randomDAG(10, prob = 0.2)")

## generate 1000 samples of DAG using standard normal error distribution
n <- 1000
d.mat <- rmvDAG(n, myDAG, errDist = "normal")

## estimate skeleton and CPDAG of given data
resU <- pcAlgo(d.mat, alpha = 0.05, corMethod = "standard")
resU
plot(resU,zvalue.lwd=TRUE)# << using the plot() method for 'pcAlgo' objects!
str(resU, max = 2)
(c.g <- compareGraphs(myDAG, resU@graph))
## CPDAG
resD <- pcAlgo(d.mat, alpha = 0.05, corMethod =
"standard",directed=TRUE)
plot(resD,zvalue.lwd=TRUE)

## plot the original DAG, the estimated skeleton and the estimated CPDAG:
op <- par(mfrow=c(3,1))
plot(myDAG, main = "original (random)DAG")
plot(resU@graph,
     main = "estimated skeleton from pcAlgo(<simulated, n =
1000>)")
plot(resD@graph,main="estimated CPDAG from pcAlgo(<simulated, n =
1000>)")
par(op)

## generate data containing severe outliers
d.mixmat <- rmvDAG(n, myDAG, errDist = "mix", mix=0.3)
## Compute "classical" and robust estimate of skeleton :
pcC <- pcAlgo(d.mixmat, 0.01, corMeth = "standard")
pcR <- pcAlgo(d.mixmat, 0.01, corMeth = "Qn")
str(pcR, max = 2)
(c.Cg <- compareGraphs(myDAG, pcC@graph))
(c.Rg <- compareGraphs(myDAG, pcR@graph))#-> (.201 0 1) much better
op <- par(mfrow=c(3,1))
  plot(myDAG, main = "original (random)DAG")
  plot(pcC)
  plot(pcR,zvalue.lwd=TRUE,lwd.max=7)
par(op)
}
\keyword{multivariate}
\keyword{models}
\keyword{graphs}
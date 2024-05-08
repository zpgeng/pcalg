### Copyright (c) 2013 - 2015  Jonas Peters  [peters@stat.math.ethz.ch]

## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
## FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <https://www.gnu.org/licenses/>.
##################################################
## exported
##################################################
lingam <- function(X, verbose = FALSE)
{
    structure(uselingam(X, verbose = verbose), class = "LINGAM")
}
setOldClass("LINGAM")

setAs("LINGAM", "amat", function(from) {
  structure(t(from$Bpruned != 0), class = "amat", type = "pag")
})


## DEPRECATED:
LINGAM <- function(X, verbose = FALSE)
## Copyright (c) 2013 - 2015  Jonas Peters  [peters@stat.math.ethz.ch]
## All rights reserved.  See the file COPYING for license terms.
{
  .Deprecated("lingam")
  res <- uselingam(X, verbose = verbose)
  list(B = res$Bpruned, Adj = t(res$Bpruned != 0))
}

##################################################
## internal
##################################################

uselingam <- function(X, verbose = FALSE) {
  t.k <- estLiNGAM(X, only.perm=TRUE, verbose=verbose)$k
  prune(t(X), t.k, verbose=verbose)
}

### MM: /u/maechler/R/MM/MISC/Speed/int-permutations.R
### --- FIXME: use e1071::permutations()

##' All permutations of integers 1:n -- as a matrix  (n!) x n
##' this a version of e1071::permutations()
##' slightly faster only for small n, __slower__ for larger n {even after MM improvements}
allPerm <- function(n) {
  p <- matrix(1L, 1L, 1L)
  if(n > 1L) for (i in 2L:n) { # 2 <= i <= n
    p <- pp <- cbind(p, i, deparse.level=0L)
    ii <- seq_len(i) # 1:i
    v <- c(ii, seq_len(i - 1L))
    for (j in 2L:i) { # 2 <= j <= i <= n
      v <- v[-1L]
      p <- rbind(p, pp[, v[ii]], deparse.level=0L)
    }
  }
  p
}

## Definition of ICA-SG

lsg <- function(m, W, X){
  # Note that m and W should be torch tensors
  p1 <- 1 / (torch_abs(torch_det(W)))^(2/3)
  S1 <- torch_zeros(ncol(X)); S2 <- torch_zeros(ncol(X))
  for (j in 1:ncol(X)) {
    for (i in 1:nrow(X)){
      flag <- torch_matmul(torch_t(W[, j]), X[i, ] - m)
      S1[j] <- torch_where(flag <= 0, 
                           S1[j] + torch_pow(flag, 2), S1[j])
      S2[j] <- torch_where(flag > 0,
                           S2[j] + torch_pow(flag, 2), S2[j])
    }
  }
  p2 <- torch_prod(torch_pow(S1, 1/3) + torch_pow(S2, 1/3))
  torch_mul(p1, p2)
}

ICA.SG <- function(X, steps){
  # center the data matrix X
  X.mean <- torch_mean(X, dim = 1)
  X.centered <- X - X.mean
  X.centered <- X.centered

  temp0 <- fastICA(X, n.comp = ncol(X))
  temp <- t(temp0$K %*% temp0$W)

  # Initialize W to conform with the dimension
  m <- torch_zeros(ncol(X), requires_grad = TRUE)
  W <- torch_tensor(temp, requires_grad = TRUE)

  optimizer <- optim_adam(list(m, W), lr = 0.1)
  threshold <- 0.1
  prev_value <- Inf

  for (i in 1:steps) {
    optimizer$zero_grad()
    value <- lsg(m, W, X.centered)
    value$backward()
    optimizer$step()

    if (abs(prev_value - value$item()) < threshold) {
      break
    }
    prev_value <- value$item()
    
    if (i %% 5 == 0) {
      cat(sprintf("Iteration %d: loss = %f\n", i, value$item()))
    }
  }
  W
}

# ICA-SN

torch_dsn <- function(x, xi=0, omega=1, alpha=0, log=FALSE)
{
  # require x, xi, omega, alpha to be torch tensors
  z <- torch_divide((x - xi), omega)
  logN <- (-torch_log(torch_sqrt(2 * pi)) - torch_log(omega) - 
             torch_pow(z, 2) / 2)
  logS <- torch_tensor(pnorm( as.numeric(alpha * z), log.p=TRUE ))
  logPDF <- logN + logS - torch_tensor(rep(log(1/2), length(x)))
  logPDF <- replace(logPDF, as(torch_abs(x) == Inf, "logical"), -Inf)
  logPDF <- replace(logPDF, as(omega <= 0, "logical"), NaN)
  out <- if(log) logPDF else toch_exp(logPDF)
  out
}

lsn <- function(m, W, alpha, X){
  # Note that m, W, and alpha should be torch tensors
  p1 <- torch_log(torch_det(W))
  den <- torch_tensor(0)
  for (j in 1:nrow(X)){
    loc <- torch_matmul(W, X[j, ])
    den <- den + sum(torch_dsn(x = loc,
                          xi = m, 
                          omega = torch_sqrt(torch_mean((X[j,] - m)^2)),
                          alpha = alpha,
                          log = TRUE))
  }
  nrow(X) * p1 + den
}

ICA.SN <- function(X, steps){
  # center the data matrix X
  X.mean <- torch_mean(X, dim = 1)
  X.centered <- X - X.mean
  X.centered <- X.centered
  
  m <- torch_zeros(ncol(X), requires_grad = TRUE)
  W <- torch_tensor(torch_diag(torch_ones(ncol(X))),
   requires_grad = TRUE)
  alpha <- torch_randn(ncol(X), requires_grad = TRUE)

  optimizer <- optim_adam(list(m, W, alpha), lr = 0.1)
  threshold <- 0.1
  prev_value <- Inf

  # Optimization loop
  for (i in 1:steps) {
    optimizer$zero_grad()
    loss <- -lsn(m, W, alpha, X.centered)
    loss$backward()
    optimizer$step()
    
    if (abs(prev_value - loss$item()) < threshold) {
      break
    }
    prev_value <- loss$item()
    
    if (i %% 5 == 0) {
      cat(sprintf("Iteration %d: loss = %f\n",
                  i, loss$item()))
    }
  }
  W
}


##' The workhorse of uselingam() and LINGAM():
##' 'only.perm' and other efficiency {t(X) !!} by Martin Maechler
estLiNGAM <- function(X, only.perm = FALSE, fastICA.tol = 1e-14,
                     pmax.nz.brute = 8, pmax.slt.brute = 8, verbose = 1, torch.steps=1000)
{
  ## --- MM: FIXME:  from  LINGAM(), we just compute  t(X)   twice, once here !!!!

    ## Using the fastICA package --> imported, see ../NAMESPACE

    ## Call the fastICA algorithm;  _FIXME_: allow all fastICA() arguments
    p <- ncol(X)
    if (verbose == 1){
      cat('Performing FastICA...\n')
      icares <- fastICA(X, n.comp = p, tol = fastICA.tol,
		      verbose = 1)
      W <- t(icares$K %*% icares$W)
    } else if (verbose == 2) {
      ## Start the ICA-SG implementation
      cat('Performing ICA-SG...\n')
      W <- ICA.SG(X, torch.steps)
      W <- as(W, "array")
      cat('Done.\n')
    } else if (verbose == 3) {
      ## Start the ICA-SN implementation
      cat('Performing ICA-SN...\n')
      W <- ICA.SN(X, torch.steps)
      W <- as(W, "array")
      cat('Done.\n')
    } else stop("You must choose a valid model,
     either 1(FastICA), 2(ICA-SG) or 3(ICA-SN).\n")

    ## [Here, we really should perform some tests to see if the 'icasig'
    ## really are independent. If they are not very independent, we should
    ## issue a warning. This is not yet implemented.]

    ## Try to permute the rows of W so that sum(1./abs(diag(W))) is minimized
    if(verbose) cat('Performing row permutation, nzdiag*() ...\n  ')
    nzd <- if (p <= pmax.nz.brute) {
               if(verbose) cat('(Small dimensionality, using brute-force method.): ')
               nzdiagbruteforce( W )
           } else {
               if(verbose) cat('(Using the Hungarian algorithm.): ')
               nzdiaghungarian( W )
           }
    Wp <- nzd$Wopt
    ## "FIXME": only making use of the 'Wopt' component of nzdiag*()
    ## rowp <- nzd$rowp
    if(verbose) cat('Done!\n')
    ## Divide each row of Wp by the diagonal element
    if(!only.perm) estdisturbancestd <- 1/diag(abs(Wp))
    Wp <- Wp/diag(Wp)

    ## Compute corresponding B
    Best <- diag(p) - Wp

    if(!only.perm) ## Estimate the constants c
      cest <- Wp %*% colMeans(X)

    ## Next, identically permute the rows and columns of B so as to get an
    ## approximately strictly lower triangular matrix
    if(verbose) cat('Performing permutation for causal order...\n  ')
    slt <- if (p <= pmax.slt.brute) {
               if(verbose) cat('(Small dimensionality, using brute-force method.): ')
               sltbruteforce( Best )
           } else {
               if(verbose) cat('(Using pruning algorithm.): ')
               sltprune( Best )
           }
    causalperm <- slt$optperm
    if(verbose) cat('Done!\n')
    if(only.perm && !verbose) # if (verbose) do the 'Bestcausal' checks below
      return(list(k = causalperm))
    Bestcausal <- slt$Bopt
    if(verbose) print(Bestcausal)

    ## Here, we report how lower triangular the result was, and in
    ## particular we issue a warning if it was not so good!
    ## MM{FIXME ?}: Warning in any case, not just if(verbose) ????
    ## --
    percentinupper <- sltscore(Bestcausal)/sum(Bestcausal^2)
    if(verbose) {
        if (percentinupper > 0.2)
            cat('WARNING: Causal B not really triangular at all!!\n')
        else if (percentinupper > 0.05)
            cat('WARNING: Causal B only somewhat triangular!\n')
        else
            cat('Causal B nicely triangular. No problems to report here.\n')
      if(only.perm)
        return(list(k = causalperm))
    }
    ## Set the upper triangular to zero
    Bestcausal[upper.tri(Bestcausal,diag=FALSE)] <- 0

    ## Finally, permute 'Bestcausal' back to the original variable
    ## ordering and rename all the variables to the way we defined them
    ## in the function definition
    icausal <- iperm(causalperm)
    ## Return the resulting list:
    list(
        B    = Bestcausal[icausal, icausal],
        stde = estdisturbancestd,
        ci   = cest,
        k    = causalperm,
        W    = W,
        pinup= percentinupper # added for firmgrowth conintegration analysis
    )
}

##' inverse permutation
##'
##' @param p
##' @author Martin Maechler
iperm <- function(p) sort.list(p, method="radix")

## Version till July 2015: is ***MUCH** slower
## iperm <- function( p ) {
##   q <- array(0,c(1,length(p)))
##   for (i in 1:length(p)) {
##     ind <- which(p==i)
##     q[i] <- ind[1]
##   }
##   q
## }

nzdiagbruteforce <- function( W ) {

  ##--------------------------------------------------------------------------
  ## Try all row permutations, find best solution
  ##--------------------------------------------------------------------------

  stopifnot((n <- nrow(W)) >= 1)

  allperms <- allPerm(n)
  I.n <- diag(n)
  bestval <- Inf
  besti <- 0
  nperms <- nrow(allperms)

  for (i in 1:nperms) {
    Pr <- I.n[, allperms[i,]] # permutation matrix
    ## FIXME{MM}: can be made faster!
    Wtilde <- Pr %*% W
    c <- nzdiagscore(Wtilde)
    if (c < bestval) {
      bestWtilde <- Wtilde
      bestval <- c
      besti <- i
    }
  }

  ## return
  list(Wopt = bestWtilde,
       rowp = iperm(allperms[besti,]))
}

nzdiaghungarian <- function( W ) {
    ## Jonas' quick additional hack to make lingam work in problems with large p
    ## (the number of dimensions from which this code is used is specified in estLiNGAM())
    n <- nrow(W)
    S <- matrix(1,n,n)/abs(W)

    ####
    ###[c,T]=hungarian(S');
    ###
    c <- as.integer(solve_LSAP(S)) # <- from CRAN pkg 'clue'

    ## Permute W to get Wopt
    Pr <- diag(n)[,c] ## FIXME{MM}: can be made faster!
    Wopt <- Pr %*% W

    ## Return the optimal permutation as well
    list(Wopt = Wopt,
         rowp = iperm(c))
}

nzdiagscore <- function(W) sum(1/diag(abs(W)))

prune <- function(X, k, method = 'resampling', # the pruning method {no other for now !}
                  prunefactor = 1, ## FIXME: argument to LINGAM !
                  npieces = 10, # <- for 'resampling
                  ## 'prunefactor' determines how easily weak connections are pruned
                  ## in the simple resampling based pruning. Zero gives no pruning
                  ## whatsoever. We use a default of 1, but remember that it is quite arbitrary
                  ## (similar to setting a significance threshold). It should probably
                  ## become an input parameter along with the data, but this is left
                  ## to some future version.
                  verbose=FALSE)
{
    ## ---------------------------------------------------------------------------
    ## Pruning
    ## ---------------------------------------------------------------------------

    if(verbose) cat('Pruning the network connections...\n')
    stopifnot(length(k) == (p <- nrow(X)))
    ndata <- ncol(X)

    ## permuting the variables to the causal order
    X.k <- X[k,] # the row-permuted X[,]
    ik <- iperm(k)
    method <- match.arg(method)
    switch(method,
           'resampling' =
      {
          ## -------------------------------------------------------------------------
          ## Pruning based on resampling: divide the data into several equally
          ## sized pieces, calculate B using covariance and QR for each piece
          ## (using the estimated causal order), and then use these multiple
          ## estimates of B to determine the mean and variance of each element.
          ## Prune the network using these.

          stopifnot(is.numeric(npieces), length(npieces) == 1,
                    npieces %% 1 == 0, 1 <= npieces, npieces <= ndata)

          ## FIXME: if ndata is *not* multiple of npieces, will *not* use remaining columns of X !!
          piecesize <- floor(ndata/npieces)

          Bpieces <- array(0,c(p,p,npieces))
          diststdpieces <- array(0,c(p,npieces))
          cpieces <- array(0,c(p,npieces))
          Bfinal <- matrix(0,p,p)
          I.p <- diag(p)

          for (i in 1:npieces) {

              ## Select subset of data
              Xp <- X.k[, ((i-1)*piecesize+1):(i*piecesize)]

              ## Remember to subtract out the mean
              Xpm <- rowMeans(Xp)
              Xp <- Xp - Xpm

              ## Calculate covariance matrix
              C <- (Xp %*% t(Xp))/ncol(Xp)

### FIXME{MM}: the epsilon (= 10^(-10)) should be *proportional* to |C|
              ## begin{HACK BY JONAS PETERS 05/2013}
              if(TRUE) {
                  ## regularization
                  C <- C + diag(10^(-10),dim(C)[1])
              }
              while(min(eigen(C, only.values=TRUE)$values) < 0)
              {
                  ## regularization
                  C <- C + diag(10^(-10),dim(C)[1])
              }
              ## end{HACK BY JONAS PETERS 05/2013}

              ## Do  QL decomposition on the inverse square root of C
              ## FIXME{MM}: solve(.) make faster -- or even faster using chol2inv(chol(.)) ?
              L <- tridecomp(invsqrtm(C), choice = 'ql', only.B = TRUE)

              ## The estimated disturbance-stds are one over the abs of the diag of L
              diag.L <- diag(L)
              newestdisturbancestd <- 1/abs(diag.L)

              ## Normalize rows of L to unit diagonal
              L <- L/diag.L

              ## Calculate corresponding B
              Bnewest <- I.p - L

              ## Also calculate constants
              cnewest <- L %*% Xpm

              ## Permute back to original variable order
              Bnewest <- Bnewest[ik, ik]
              newestdisturbancestd <- newestdisturbancestd[ik]
              cnewest <- cnewest[ik]

              ## Save results
              Bpieces[,,i] <- Bnewest
              diststdpieces[,i] <- newestdisturbancestd
              cpieces[,i] <- cnewest
          }

          for (i in 1:p) {
              Bp.i <- Bpieces[i,,]
              for (j in 1:p) {
                  themean <- mean(Bp.i[j,])
                  thestd  <- sd  (Bp.i[j,])
                  Bfinal[i,j] <- if (abs(themean) < prunefactor*thestd) 0 else themean
              }
          }

        stde   <- rowMeans(diststdpieces)
        cfinal <- rowMeans(cpieces)
      },
      'olsboot' = {
	  stop(gettextf("Method '%s' not implemented yet!", method), domain=NA)
      },
      'wald' = {
	  stop(gettextf("Method '%s' not implemented yet!", method), domain=NA)
      },
      'bonferroni' = {
	  stop(gettextf("Method '%s' not implemented yet!", method), domain=NA)
      },
      'hochberg' = {
	  stop(gettextf("Method '%s' not implemented yet!", method), domain=NA)
      },
      'modelfit' = {
	  stop(gettextf("Method '%s' not implemented yet!", method), domain=NA)
      }
      )## end{ switch( method, ..) }

    if(verbose) cat('Done!\n')

    ## Return the result
    list(Bpruned = Bfinal, stde = stde, ci = cfinal)
}

## SLT = Strict Lower Triangularity


sltbruteforce <- function( B ) {

  ##--------------------------------------------------------------------------
  ## Try all row permutations, find best solution
  ##--------------------------------------------------------------------------

  n <- nrow(B)

  bestval <- Inf
  besti <- 0
  allperms <- allPerm(n)
  nperms <- nrow(allperms)

  for (i in 1:nperms) {
    p.i <- allperms[i,]
    Btilde <- B[p.i, p.i] # permuted B
    c <- sltscore(Btilde)
    if (c < bestval) {
      bestBtilde <- Btilde
      bestval <- c
      besti <- i
    }
  }

  ## return
  list(Bopt = bestBtilde,
       optperm = allperms[besti,])
}

sltprune <- function(B) {
    ## Hack of JONAS PETERS 2013
    n <- nrow(B)
    ##[y,ind] = sort(abs(B(:)));
    ind <- sort.list(abs(B))

    for(i in ((n*(n+1)/2):(n*n))) {
        Bi <- B ## Bi := B, with the i smallest (in absolute value) coefficients to zero
        Bi[ind[1:i]] <- 0
        ## Try to do permutation
        p <- slttestperm( Bi )

        ## If we succeeded, then we're done!
        if(any(p != 0)) {
            Bopt <- B[p,p, drop=FALSE]
            break
        }
        ## ...else we continue, setting one more to zero!
    }
    ## return :
    list(Bopt = Bopt, optperm = p)
}

##' Measuring how close B is to SLT = Strict Lower Triangular
sltscore <- function (B) sum((B[upper.tri(B,diag=TRUE)])^2)

slttestperm <- function(B, rowS.tol = 1e-12)
{
    ## Hack of JONAS PETERS 2013;  tweaks: MM, 2015-07
    ##
    ## slttestperm - tests if we can permute B to strict lower triangularity
    ##
    ## If we can, then we return the permutation in p, otherwise p=0.
    ##

    ## Dimensionality of the problem
    stopifnot((n <- nrow(B)) >= 1, rowS.tol >= 0)

    ## This will hold the permutation
    p <- integer(0)

    ## Remaining nodes
    remnodes <- 1:n

    ## Remaining B, take absolute value now for convenience
    Brem <- abs(B)
    ## Select nodes one-by-one
    for(ii in 1:n)
    {
        ## Find the row with all zeros
        ## therow = find(sum(Brem,2)<1e-12)
        rowS <- if(length(Brem) > 1) rowSums(Brem) else Brem
        therow <- which(rowS < rowS.tol)

        ## If empty, return 0
        if(length(therow) == 0L)
            return(0L)
        ## If we made it to the end, then great!
        if(ii == n)
            return(c(p, remnodes))
        ## If more than one, arbitrarily select the first
        therow <- therow[1]
        ## Take out that row and that column
	Brem <- Brem[-therow, -therow, drop=FALSE]
        ### CHECK!!!!

        ## Update remaining nodes
        p <- c(p,remnodes[therow])
	remnodes <- remnodes[-therow]
    }
    stop("the program flow should never get here [please report!]")
}

sqrtm <- function( A ) {
  e <- eigen(A)
  V <- e$vectors
  ## return B =
  ## MM: FIXME (diag mult)
  V %*% diag(sqrt(e$values)) %*% t(V)
}
##' @title Inverse of Matrix Square Root, \eqn{A^{-1/2}}
##' @param A square matrix, here must be semi positive definite
##' @return \eqn{A^{-1/2}} via Eigen decomposition
##' @author Martin Maechler
invsqrtm <- function( A ) {
  e <- eigen(A)
  V <- e$vectors
  lam.m.5 <- 1/sqrt(e$values) # \lambda^{-1/2}
  ## return B =
  ## MM: FIXME (diag mult)
  ## V %*% diag(lam.m.5) %*% t(V)
  tcrossprod(V * rep(lam.m.5, each=nrow(V)), V)
}


tridecomp <- function (W, choice='qr', only.B = FALSE) {

  ## SYNTAX:
  ## res <- tridecomp( W, choice )
  ## QR, RQ, QL, or LQ decomposition specified by
  ## choice = 'qr', 'rq', 'ql', or 'lq' respectively
  ##
  ## if(only.B) return the 2nd matrix only,
  ## otherwise list(A = the  first matrix,
  ##                B = the second matrix)
  ##
  ## Based on MATLAB code kindly provided by Matthias Bethge
  ## Adapted for R by Patrik Hoyer -- and improved by Martin Maechler

  stopifnot(1 <= (m <- nrow(W)), 1 <= (n <- ncol(W)))
  ## FIXME{MM}: faster
  ## Jm and Jn are simple "revert order" permutation matrices
  Jm <- matrix(0,m,m)
  Jm[m:1,] <- diag(m)
  Jn <- matrix(0,n,n)
  Jn[n:1,] <- diag(n)

  switch(choice,
	 'qr' = {
	   r <- qr(W)
	   if(only.B) qr.R(r)
	   else
	     list(A = qr.Q(r),
		  B = qr.R(r))
	 },
	 'lq' = {
	   r <- qr(t(W))
	   if(only.B) t(qr.Q(r))
	   else
	     list(A = t(qr.R(r)),
		  B = t(qr.Q(r)))
	 },
	 'ql' = {
	   r <- qr(Jm %*% W %*% Jn)
	   if(only.B) Jm %*% qr.R(r) %*% Jn
	   else
	     list(A = Jm %*% qr.Q(r) %*% Jm,
		  B = Jm %*% qr.R(r) %*% Jn)
	 },
	 'rq' =  {
	   r <- qr(Jn %*% t(W) %*% Jm)
	   if(only.B) t(Jn %*% qr.Q(r) %*% Jn)
	   else
	     list(A = t(Jn %*% qr.R(r) %*% Jm),
		  B = t(Jn %*% qr.Q(r) %*% Jn))
	 },
	 stop("invalid 'choice': ", choice))
}

## Local Variables:
## eval: (ess-set-style 'DEFAULT 'quiet)
## delete-old-versions: never
## End:


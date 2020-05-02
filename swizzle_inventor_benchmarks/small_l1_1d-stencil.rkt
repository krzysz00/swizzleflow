#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
 | Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 |
 | Redistribution and use in source and binary forms, with or without 
 | modification, are permitted provided that the following conditions are met:
 |
 | 1. Redistributions of source code must retain the above copyright notice, 
 | this list of conditions and the following disclaimer.
 |
 | 2. Redistributions in binary form must reproduce the above copyright notice, 
 | this list of conditions and the following disclaimer in the documentation 
 | and/or other materials provided with the distribution.
 |
 | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 | AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 | IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 | ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 | LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 | CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 | SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 | INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 | CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 | ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 | POSSIBILITY OF SUCH DAMAGE.
 |#

#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define n-block 2)
(define /3 (lambda (x) (/ x 3)))

;; Create input and output matrices.
(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 2 warpSize))
  (define I-sizes (x-y-z (* 2 block-size)))
  (define O-sizes (- I-sizes 2))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix O-sizes))
  (define O* (create-matrix O-sizes))
  (values block-size I-sizes O-sizes I O O*))

;; Run sequential program spec and GPU kernel kernel, and compare their outputs.
(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes O-sizes I O O*)
    (create-IO w))

  (spec I O O-sizes)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* I-sizes O-sizes)
  ;(acc-print O*)
  (acc-equal? O O*)
  )

;; Sequential program spec
(define (stencil-1d-spec I O o-sizes)
  (for ([i (get-x o-sizes)])
    (let ([o (create-accumulator (list +) /3)])
      (for ([j 3])
        (accumulate o (get I (+ i j))))
      (set O i o))))

;; Complete kernel
(define (stencil-1d threadId blockID blockDim I O I-sizes O-sizes)
  (define I-cached (create-matrix-local (x-y-z 2)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* blockID blockDim) (* warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z 1)
                 (+ (* blockID blockDim) (* warpID warpSize))
                 (x-y-z (+ warpSize 2)) #f #:round 2)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator (list +) /3 blockDim))
  (for ([i 3])
    (let* ([index (ite (< localId i) 1 0)]
           [lane (+ i localId)]
           [x (shfl (get I-cached index) lane)])
      ;(pretty-display `(lane ,i ,localId ,lane))
      (accumulate o x)
      ))
  (reg-to-global o O gid)
  )

;; Kernel sketch
(define (stencil-1d-sketch threadId blockID blockDim I O I-sizes O-sizes)
  (define I-cached (create-matrix-local (x-y-z 2)))
  (define gid (+ (* blockID blockDim) threadId))
  (define localId (get-idInWarp threadId))
  (define offset (- gid localId))
  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (+ warpSize 2)) #f #:round 2)

  (define o (create-accumulator (list +) /3 blockDim))

  (for/bounded ([i 3])
    (let* ([index (ite (?cond-easy (@dup i) localId) (@dup 0) (@dup 1))]
           [lane (?sw-xform-easy localId warpSize
                            i warpSize)]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x #:pred (?cond-easy localId (@dup i))) ; (?cond-easy localId (@dup i))
      ))
  (reg-to-global o O gid)
  )

;; Check correctness of a complete kernel against a spec.
(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size stencil-1d-spec stencil-1d-sketch w)])
      (pretty-display `(test ,w ,ret))))
  )
;(test)

;; Synthesize a kernel sketch given a spec.
(define (synthesis)
  (pretty-display "solving...")
  (define sol
    (time (solve
           (assert (andmap
                    (lambda (w) (run-with-warp-size stencil-1d-spec stencil-1d-sketch w))
                    (list 32))))))
  (print-forms sol)
  )
(define t0 (current-seconds))
(synthesis)
(define t1 (current-seconds))
(- t1 t0)

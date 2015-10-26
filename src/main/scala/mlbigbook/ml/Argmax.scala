package mlbigbook.ml

import mlbigbook.data.DataClass

import scala.reflect.ClassTag

/**
 * Generic algorithm for finding the maximal argument. Uses the `Val`
 * typeclass as evidence of an argument's value.
 */
object Argmax {

  val emptyError =
    new IllegalArgumentException("Cannot call `Argmax.apply` on empty elements.")

  /**
   * Finds the maximal argument of `elements` in linear time. Uses the `Val`
   * typeclass as evidence of an argument's value.
   *
   * throws IllegalArgumentException Iff `elements` is empty.
   */
  def apply[B: ClassTag](elements: DataClass[B])(implicit ev: Val[B]): B =
    if (elements isEmpty)
      throw emptyError

    else
      elements
        .reduce[B] {
          case (a, b) =>
            if (ev.valueOf(a) > ev.valueOf(b))
              a
            else
              b
        }
}

/**
 * Typeclass for giving a value to a type `X`.
 */
trait Val[-X] {
  def valueOf(a: X): Double
}

object TupleVal1 {
  def apply[X] = new TupleVal1[X] {}
}

/**
 * Value for a tuple of (Double, X) type.
 */
trait TupleVal1[X] extends Val[(Double, X)] {
  override def valueOf(a: (Double, X)): Double =
    a._1
}

object TupleVal2 {
  def apply[X] = new TupleVal2[X] {}
}

/**
 * Value for a tuple of (X, Double) type.
 */
trait TupleVal2[X] extends Val[(X, Double)] {
  override def valueOf(a: (X, Double)): Double =
    a._2
}
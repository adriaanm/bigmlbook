package mlbigbook.ml

import fif.Data
import Data.ops._

import scala.language.higherKinds

object Counting {

  def empty[T, N: Numeric]: Map[T, N] =
    Map.empty[T, N]

  def increment[T, N: Numeric](map: Map[T, N], e: T): Map[T, N] =
    increment(map, e, implicitly[Numeric[N]].one)

  def increment[T, N: Numeric](map: Map[T, N], e: T, times: N): Map[T, N] =
    if (map contains e)
      (map - e) + (e -> implicitly[Numeric[N]].plus(map(e), times))
    else
      map + (e -> times)

  def count[T, N: Numeric, D[_]: Data](data: D[T]): Map[T, N] =
    count(empty, data)

  def count[T, N: Numeric, D[_]: Data](existingMap: Map[T, N], data: D[T]): Map[T, N] =
    data.aggregate(existingMap)(increment[T, N], combine[T, N])

  /**
   * Combines two maps. If maps m1 and m2 both have key k, then the resulting
   * map will have m1(k) + m2(k) for the value of k.
   */
  def combine[T, N: Numeric](m1: Map[T, N], m2: Map[T, N]): Map[T, N] = {
    val (smaller, larger) =
      if (m1.size < m2.size)
        (m1, m2)
      else
        (m2, m1)

    smaller.foldLeft(larger) {
      case (aggmap, (k, v)) =>
        aggmap.get(k) match {

          case Some(existing) =>
            (aggmap - k) + (k -> implicitly[Numeric[N]].plus(existing, v))

          case None =>
            aggmap + (k -> v)
        }
    }
  }

}
package mlbigbook.ml

import breeze.linalg.{ DenseVector, Vector }
import breeze.linalg.support.CanMapValues
import fif.Data
import mlbigbook.data.DataClass
import mlbigbook.math.{ VectorOpsT, NumericConversion, OnlineMeanVariance, NumericX }
import simulacrum.typeclass

import scala.annotation.tailrec
import scala.language.higherKinds
import scala.reflect.ClassTag

object FeatureVectorSupport {

  sealed trait Value
  case class Categorical(v: String) extends Value
  case class Real(v: Double) extends Value

  type FeatVec = Seq[Value]

  case class FeatureSpace(
    features:      Seq[String],
    isCategorical: Seq[Boolean]
  )

  type Discretized = Seq[String]

  type IndexRange = Seq[Int]

  case class DiscretizedFeatureSpace(
    original:                      FeatureSpace,
    expandedFeatures:              Seq[String],
    originalIndex2ExpandedIndices: Seq[IndexRange]
  )

}

trait DecisionTree {

  /**
   * Abstract type. A stand-in for the overall decision type of an entire
   * decision tree. Common cases include String (e.g.
   */
  type Decision

  /**
   * Abstract type. A stand-in for the feature vector that the decision tree
   * uses during learning and classification.
   */
  type FeatureVector

  /**
   * The children of a parent node. Must always be non-empty.
   */
  final type Children = Seq[Node]

  /**
   * A parent node's feature test. Functions of this type accept a non-empty
   * sequence of children and a feature vector. The test will inspect values
   * from the vector to determine which of the children to output.
   */
  final type Test = (Children, FeatureVector) => Node

  /**
   * A node of a decision tree. By itself, a Node instance is a fully
   * functional decision tree. One may grow a decision tree from a single node,
   * use an existing decision tree as a subtree of another, or prune a decision
   * tree by selectively removing nodes.
   *
   * The Node abstract data type has two concrete instantiations:
   * Parent and Leaf.
   */
  sealed trait Node

  /**
   * A Parent is a Node sub-type that makes up a decision tree. Parents contain
   * one or more child nodes and a Test function. The Test function is used in
   * the decision making process. When presented with an input feature vector,
   * one uses a Parent node's Test function to select from one of its own
   * children.
   */
  case class Parent(t: Test, c: Children) extends Node

  /**
   * A Leaf is a Node sub-type that makes up a decision tree. Leaves are the
   * final decision making aspect of a decision tree. The decision process
   * stops once a Leaf is encountered. At such a point, the Decision instance
   * contained within the Leaf is returned as the decision tree's response.
   */
  case class Leaf(d: Decision) extends Node

  /**
   * The function type for making decisions using a decision tree node.
   */
  type Decider = Node => FeatureVector => Decision

  /**
   * Implementation of the decision making process using a decision tree.
   * Is efficient and guaranteed to not incur a stack overflow.
   */
  val decide: Decider =
    decisionTree => featureVector => {

        @tailrec def descend(n: Node): Decision =
          n match {

            case Parent(test, children) =>
              descend(test(children, featureVector))

            case Leaf(d) =>
              d
          }

      descend(decisionTree)
    }
}

object Id3LearningSimpleFv {

  import fif.Data.ops._
  import FeatureVectorSupport._

  /*

    (1) Calculate the entropy of every attribute using the data set S.

    (2) Split the set S into subsets using the attribute for which entropy is
        minimum (or, equivalently, information gain is maximum).

    (3) Make a decision tree node containing that attribute.

    (4) Recurse on subsets using remaining attributes.

   */

  def apply[D[_]: Data, T <: DecisionTree { type FeatureVector = FeatVec; type Decision = String }](
    data: D[(FeatVec, String)]
  )(
    implicit
    fs: FeatureSpace
  ): T#Node = ???

  def apply[D[_]: Data, T <: DecisionTree { type FeatureVector = Discretized; type Decision = String }](
    data: D[(Discretized, String)]
  )(
    implicit
    efs: DiscretizedFeatureSpace
  ): T#Node = ???

}

object InformationSimpleFv {

  import fif.Data.ops._
  import FeatureVectorSupport._

  def discretize[D[_]: Data](
    data: D[FeatVec]
  )(
    implicit
    fs: FeatureSpace
  ): (DiscretizedFeatureSpace, D[Discretized]) = {
    ???
  }

  def entropy[D[_]: Data, FV](
    data: D[FV]
  )(
    implicit
    fs:   FeatureSpace,
    isFv: FV => FeatVec
  ): Seq[Double] = {

    /*

      for each feature
        - count events

      for each feature
        for each event in feature:
          - calculate P(event) ==> # events / total
          - entropy(feature)   ==> - sum( p(event) * log_2( p(event) ) )

     */

    //    val feat2event2count = Map.empty[String, Map[String, ]]

    ???

  }

}

trait Id3Learning {

  import fif.Data.ops._

  /*
   (1) Calculate the entropy of every attribute using the data set S.

   (2) Split the set S into subsets using the attribute for which entropy is
       minimum (or, equivalently, information gain is maximum).

   (3) Make a decision tree node containing that attribute.

   (4) Recurse on subsets using remaining attributes.
   */

  type Entropy
  implicit def entIsNum: NumericX[Entropy]

  trait FeatureSpace[N, V[_] <: Vector[_]] {

    implicit def numConv: NumericConversion[N]
    implicit def vecOps: VectorOpsT[N, V]

    def size: Int
    def nameOf(index: Int): Option[String]
    def range: IndexedSeq[Boolean]
    def zero: Boolean
  }

  //  trait ContinousEntropy[C] {
  //    def continuous[D[_] : Data, N: NumericConversion, V[_] <: Vector[_]](c: C)(data: D[V[N]])(implicit fs: FeatureSpace[N,V]): V[Double]
  //  }

  //  object GaussianEstimatedEntropy {
  //
  //    private[this] val const = math.sqrt(2.0 * math.Pi * math.E)
  //
  //    def continuous[D[_] : Data, N : NumericConversion, V[_] <: Vector[_]](data: D[V[N]])(implicit fs: FeatureSpace[N, V]) = {
  //
  //      import fs._
  //      val Stats(_, _, variance) = OnlineMeanVariance.batch(data.asInstanceOf[DataClass[V[N]]])
  //
  //      variance.map { sigmaSq =>
  //          val sigma = math.sqrt(implicitly[NumericConversion[N]].numeric.toDouble(sigmaSq))
  //          math.log(sigma * const)
  //        }
  //    }
  //  }

  trait ContinousEntropy {

    type N
    type V[_] <: Vector[_]
    type D[_]

    implicit def d: Data[D]
    implicit val fs: FeatureSpace[N, V]

    def continuous(data: D[V[N]]): V[N]

    object CanMapValuesSupport {

      def apply[B]: CanMapValues[V[N], N, B, V[B]] =
        new Foo[B] {}

      trait Foo[B] extends CanMapValues[V[N], N, B, V[B]] {

        /**Maps all key-value pairs from the given collection. */
        def map(from: V[N], fn: (N => B)): V[B] = ???
        //          from.values.map[V[_], B, V[B]](fn)

        /**Maps all active key-value pairs from the given collection. */
        def mapActive(from: V[N], fn: (N => B)): V[B] = ???
      }
    }
  }

  @typeclass trait VectorHofs[V[_] <: Vector[_]] {

    def map[A: Numeric, B: ClassTag](v: V[A])(f: A => B): V[B]

    def foldLeft[A: Numeric, B: Numeric](v: V[A])(zero: B)(comb: (B, A) => B): B

    def foldRight[A: Numeric, B: Numeric](v: V[A])(zero: B)(comb: (A, B) => B): B

    def reduce[A: Numeric](v: V[A])(r: (A, A) => A): A

  }

  object ImplicitVectorHofs {

    def apply[V[_] <: Vector[_]: VectorHofs]: VectorHofs[V] =
      implicitly[VectorHofs[V]]

    implicit object DenseVectorHof extends VectorHofs[DenseVector] {

      override def map[A: Numeric, B: ClassTag](v: DenseVector[A])(f: (A) => B): DenseVector[B] = {

        val size = v.length
        val arr = new Array[B](size)
        val d = v.data
        val stride = v.stride

        var i = 0
        var j = v.offset

        while (i < size) {
          arr(i) = f(d(j))
          i += 1
          j += stride
        }

        new DenseVector[B](arr)
      }

      override def foldLeft[A: Numeric, B: Numeric](v: DenseVector[A])(zero: B)(comb: (B, A) => B): B = {
        // The entire definition is equivalent to:
        // v.valuesIterator.foldLeft(zero)(comb)
        val size = v.length
        val d = v.data
        val stride = v.stride

        var accum = zero
        var i = 0
        var j = v.offset

        while (i < size) {
          accum = comb(accum, d(j))
          i += 1
          j += stride
        }

        accum
      }

      override def foldRight[A: Numeric, B: Numeric](v: DenseVector[A])(zero: B)(comb: (A, B) => B): B = {
        // The entire definition is equivalent to:
        // v.valuesIterator.foldRight(zero)(comb)
        val size = v.length
        val d = v.data
        val stride = v.stride

        var accum = zero
        var i = size
        var j = i - v.offset

        while (i >= 0) {
          accum = comb(d(j), accum)
          i -= 1
          j -= stride
        }

        accum
      }

      override def reduce[A: Numeric](v: DenseVector[A])(r: (A, A) => A): A = {
        // The entire definition is equivalent to:
        // v.valuesIterator.reduce(r)
        // TODO Implement more efficient imperative reduce for DenseVector s!
        v.valuesIterator.reduce(r)
      }

    }

  }

  // TODO delete me
  def test[V[_] <: Vector[_]: VectorHofs](v: V[Double]): V[Double] = {
    ImplicitVectorHofs[V].map(v)(value => value * 2.0)
  }

  trait GaussianEstimatedEntropy extends ContinousEntropy {

    private[this] val const = math.sqrt(2.0 * math.Pi * math.E)

    override def continuous(data: D[V[N]]) = {
      import Data.ops._

      import fs._
      val Stats(_, _, variance) = OnlineMeanVariance.batch(data.asInstanceOf[DataClass[V[N]]])

      ???
      //      variance.map { sigmaSq =>
      //        val sigma = math.sqrt(implicitly[NumericConversion[N]].numeric.toDouble(sigmaSq.asInstanceOf[N]))
      //        math.log(sigma * const)
      //      }(CanMapValuesSupport[Double])
    }

  }

  trait GaussianEstimatedEntropy__ {

    type N
    type V[_] <: Vector[_]
    type D[_]

    implicit def d: Data[D]
    implicit val fs: FeatureSpace[N, V]
    import fs._

    private[this] val const = math.sqrt(2.0 * math.Pi * math.E)

    def continuous(data: D[V[N]]) = {
      import Data.ops._

      import fs._
      val Stats(_, _, variance) = OnlineMeanVariance.batch(data.asInstanceOf[DataClass[V[N]]])

      variance.map { sigmaSq =>
        val sigma = math.sqrt(implicitly[NumericConversion[N]].numeric.toDouble(sigmaSq.asInstanceOf[N]))
        math.log(sigma * const)
      }
    }
  }

  object Implicits {
    //    implicit object GaussIsCont extends ContinousEntropy[GaussianEstimatedEntropy] {
    //
    //    }
  }

}

object OptionSeqDsl {

  implicit class GetOrEmptySeq[V](val x: Option[Seq[V]]) extends AnyVal {
    def getOrEmpty: Seq[V] =
      x.getOrElse(Seq.empty[V])
  }

}

object BinaryTreeExplore {

  import OptionSeqDsl._

  sealed trait Node[V]
  case class Parent[V](left: Option[Node[V]], item: V, right: Option[Node[V]]) extends Node[V]
  case class Leaf[V](item: V) extends Node[V]

  //
  // Traversals
  //

  type Traverser[V] = Node[V] => Seq[V]

  // From Wikipedia:
  /*
		Pre-order
			Display the data part of root element (or current element)
			Traverse the left subtree by recursively calling the pre-order function.
			Traverse the right subtree by recursively calling the pre-order function.
	*/

  def preOrder[V]: Traverser[V] = {

    case Parent(left, item, right) =>
      item +: (left.map(preOrder).getOrEmpty ++ right.map(preOrder).getOrEmpty)

    case Leaf(item) =>
      Seq(item)
  }

  // From Wikipedia:
  /*
		In-order (symmetric)[edit]
			Traverse the left subtree by recursively calling the in-order function
			Display the data part of root element (or current element)
			Traverse the right subtree by recursively calling the in-order function
  */

  def inOrder[V]: Traverser[V] = {

    case Parent(left, item, right) =>
      (left.map(inOrder).getOrEmpty :+ item) ++ right.map(inOrder).getOrEmpty

    case Leaf(item) =>
      Seq(item)
  }

  // From Wikipedia:
  /*
		Post-order[edit]
			Traverse the left subtree by recursively calling the post-order function.
			Traverse the right subtree by recursively calling the post-order function.
			Display the data part of root element (or current element).
	*/

  def postOrder[V]: Traverser[V] = {

    case Parent(left, item, right) =>
      left.map(postOrder).getOrEmpty ++ right.map(postOrder).getOrEmpty :+ item

    case Leaf(item) =>
      Seq(item)
  }

}

object GenericTreeExplore {

  sealed trait Node[V]
  case class Parent[V](children: Seq[Node[V]], item: V) extends Node[V]
  case class Leaf[V](item: V) extends Node[V]

  //
  // Traversals
  //

  type Traverser[V] = Node[V] => Seq[V]

  def preOrder[V]: Traverser[V] = {

    case Parent(children, item) =>
      item +: children.flatMap(preOrder)

    case Leaf(item) =>
      Seq(item)
  }

  def postOrder[V]: Traverser[V] = {

    case Parent(children, item) =>
      children.flatMap(postOrder) :+ item

    case Leaf(item) =>
      Seq(item)
  }

  // NOTE: There is no in-order.
  // For more than 2 nodes, an in-order traversal is ambigious. Where do we "insert" the node?
  // From the n children, which one do we choose? Pick some k. What if k is less than the # of
  // children? If it changes every time, then it's totally arbitrary and inconsistent.
  // Only makes sense to have either (1) item before everything or (2) item after everything.
  // These are the only traversals that will have consistent ordering.

}
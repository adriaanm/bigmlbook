package mlbigbook.ml

import mlbigbook.math.NumericX

object FeatureVectorSupport {
  final type FeatVec = breeze.linalg.Vector[Value]
  final type Value = Double
}

trait DecisionTree {

  import FeatureVectorSupport._

  type Decision
  final type Children = Seq[Node]
  final type Test = Children => FeatVec => Node

  sealed trait Node
  case class Parent(t: Test, c: Children) extends Node
  case class Leaf(d: Decision) extends Node

  type Decider = Node => FeatVec => Decision
  def decide: Decider =
    decisionTree => featureVector => {

        def descend: Node => Decision = {

          case Parent(test, children) =>
            descend(test(children)(featureVector))

          case Leaf(d) =>
            d
        }

      descend(decisionTree)
    }
}

trait Id3Learning {
  /*
  Calculate the entropy of every attribute using the data set S
  Split the set S into subsets using the attribute for which entropy is minimum (or, equivalently, information gain is maximum)
  Make a decision tree node containing that attribute
  Recurse on subsets using remaining attributes.
   */

  type Entropy
  implicit def entIsNum: NumericX[Entropy]

//  def entropyOf[D:]

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
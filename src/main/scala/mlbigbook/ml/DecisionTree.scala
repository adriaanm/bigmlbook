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

}

object InformationSimpleFv {

  import fif.Data.ops._
  import FeatureVectorSupport._
  import OnlineMeanVariance._

  def entropy[D[_]: Data, FV](
    data: D[FV]
  )(
    implicit
    fs:   FeatureSpace,
    isFv: FV => FeatVec
  ): Seq[Double] = {

    //
    //
    //
    // DO CONTINOUS VARIABLES FIRST
    //
    //
    //

    val (realIndices, catIndices) =
      fs.isCategorical
        .zipWithIndex
        .foldLeft((Seq.empty[Int], Seq.empty[Int])) {
          case ((ri, ci), (isCategory, index)) =>
            if (isCategory)
              (ri, ci :+ index)
            else
              (ri :+ index, ci)
        }

    val realFeatureNames =
      realIndices.map { index => fs.features(index) }

    val catFeatureNames =
      catIndices.map { index => fs.features(index) }

    /*

    Show first:

    Example of doing this for one case:

     val realIndices =
      fs.isCategorical
        .zipWithIndex
        .flatMap {
          case (isCategory, index) =>
            if (isCategory)
              None
            else
              Some(index)
        }


    // analogus case

    val catIndices =
      fs.isCategorical
        .zipWithIndex
        .flatMap {
          case (isCategory, index) =>
            if(isCategory)
              Some(index)
            else
              None
        }

    // we're reapeating ourselves!
    // maybe put into a function and apply twice?
    // ...
    // let's give that a few seconds' thought
    // ...
    // are we really going to _reuse_ this function?
    // no
    // it's also not as efficient: we're going through the feature space twice
    // what if we fold our way through the space, accumulating both sequences?
    //
    // <enter actual solution>

     */

    val realOnly: D[DenseVector[Double]] =
      data.map { fv =>
        val realValues =
          realIndices.map { index =>
            fv(index) match {

              case Real(v) =>
                v

              case Categorical(_) =>
                throw new IllegalStateException(
                  s"Violation of FeatureSpace contract: feature at index $index is categorical, expecting real"
                )
            }
          }
            .toArray

        DenseVector(realValues)
      }

    import NumericConversion.Implicits._
    import VectorOpsT.Implicits._

    val statsForAllRealFeatures = OnlineMeanVariance.batch[D, Double, DenseVector](realOnly)

    import MathOps.Implicits._
    val gf = GaussianFactory[Double]

    val realFeat2estMeanVar: Map[String, GaussianFactory[Double]#Gaussian] =
      realIndices.map { index =>
        val g = new gf.Gaussian(
          mean = statsForAllRealFeatures.mean(index),
          variance = statsForAllRealFeatures.variance(index),
          stddev = math.sqrt(statsForAllRealFeatures.variance(index))
        )
        (fs.features(index), g)
      }
        .toMap

    //
    //
    //
    // DO CATEGORICAL VARIABLES SECOND
    //
    //
    //

    val categoricalOnly: D[Seq[String]] =
      data.map { fv =>

        catIndices.map { index =>

          fv(index) match {

            case Categorical(v) =>
              v

            case Real(_) =>
              throw new IllegalStateException(
                s"Violation of FeatureSpace contract: feature at index $index is real, expecting categorical"
              )
          }
        }
      }

    //
    //
    //
    // CALCULATE ENTROPY FOR ALL FEATURES
    //
    //
    //

    /*
      Strategy for discrete variables:

      for each feature
        - count events

      for each feature
        for each event in feature:
          - calculate P(event) ==> # events / total
          - entropy(feature)   ==> - sum( p(event) * log_2( p(event) ) )
     */

    val catFeat2event2count =
      categoricalOnly.aggregate(Map.empty[String, Map[String, Long]])(
        {
          case (feat2event2count, featureValues) =>
            featureValues.zip(realFeatureNames)
              .foldLeft(feat2event2count) {
                case (m, (value, name)) =>
                  Counting.incrementNested(
                    m,
                    name,
                    value
                  )
              }
        },
        Counting.combineNested[String, String, Long]
      )

    val catFeat2Entropy =
      catFeat2event2count.map {
        case (feature, event2count) =>

          val entropyOfFeature = {
            val totalEventCount = event2count.values.sum.toDouble
            -event2count.foldLeft(0.0) {
              case (sum, (_, count)) =>
                val probabilityEvent = count / totalEventCount
                probabilityEvent * logBase2(probabilityEvent)
            }
          }

          (feature, entropyOfFeature)
      }

    /*
      Strategy for continuous variables:

        (1) Calculate mean & variance for each continous variable.
        (2) Construct a gaussian with ^^.
        (3) Calculate entropy of each estimated gaussian.
     */

    ???
  }

  val logBase2 = logBaseX(2.0) _

  def logBaseX(base: Double)(value: Double): Double =
    math.log(value) / math.log(base)

}

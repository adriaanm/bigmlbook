package mlbigbook.ml

import breeze.linalg.DenseVector
import fif.{TravData, Data}
import mlbigbook.math.{ VectorOpsT, NumericConversion, OnlineMeanVariance }

import scala.annotation.tailrec
import scala.language.{ postfixOps, higherKinds }

object FeatureVectorSupport {

  sealed trait Value
  case class Categorical(v: String) extends Value
  case class Real(v: Double) extends Value

  type FeatVec = Seq[Value]

  case class FeatureSpace(
    features:      Seq[String],
    isCategorical: Seq[Boolean],
    feat2index:    Map[String, Int]
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


    ===============


    ID3 (Examples, Target_Attribute, Attributes)
    Create a root node for the tree
    If all examples are positive, Return the single-node tree Root, with label = +.
    If all examples are negative, Return the single-node tree Root, with label = -.
    If number of predicting attributes is empty, then Return the single node tree Root,
    with label = most common value of the target attribute in the examples.
    Otherwise Begin
        A ← The Attribute that best classifies examples.
        Decision Tree attribute for Root = A.
        For each possible value, vi, of A,
            Add a new tree branch below Root, corresponding to the test A = vi.
            Let Examples(vi) be the subset of examples that have the value vi for A
            If Examples(vi) is empty
                Then below this new branch add a leaf node with label = most common target value in the examples
            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
    End
    Return Root

   */

  def apply[D[_]: Data, T <: DecisionTree { type FeatureVector = FeatVec; type Decision = Boolean }](
    dtModule: T,
    data: D[(FeatVec, Boolean)]
  )(
    implicit
    fs: FeatureSpace
  ): Option[T#Node] = {
    implicit val _ = dtModule
    learn(data, 0 until fs.features.size)
  }

  protected def learn[D[_]: Data, T <: DecisionTree { type FeatureVector = FeatVec; type Decision = Boolean }](
    data:         D[(FeatVec, Boolean)],
    featuresLeft: Seq[Int]
  )(
    implicit
    fs: FeatureSpace,
    dtModule: T
  ): Option[T#Node] =

    if (data isEmpty)
      None

    else
      Some {

        val (nPos, nNeg) =
          data.aggregate((0l, 0l))(
            {
              case ((nP, nN), (_, label)) =>
                if (label)
                  (nP + 1l, nN)
                else
                  (nP, nN + 1l)
            },
            {
              case ((nP1, nN1), (nP2, nN2)) =>
                (nP1 + nP2, nN1 + nN2)
            }
          )

        if (featuresLeft isEmpty) {
          if (nPos > nNeg)
            new dtModule.Leaf(true)
          else
            new dtModule.Leaf(false)

        } else {

          (nPos, nNeg) match {

            case (0l, nonZero) =>
              new dtModule.Leaf(false)

            case (nonZero, 0l) =>
              new dtModule.Leaf(true)

            case (_, _) =>

              val entropyOfFeatures = {
                implicit val _ = (x: (FeatVec, Boolean)) => x._1
                InformationSimpleFv.entropy(data)
              }

              val indexOfMinEntropyFeature = {
                implicit val v = TupleVal1[Int]
                implicit val td = TravData
                Argmin(entropyOfFeatures.zipWithIndex.toTraversable)
              }

              // partition data according to the discrete values of each

              ???
          }
        }
      }

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

    //
    //
    //
    // DO CONTINUOUS VARIABLES
    //
    //
    //

    /*
      Strategy for continuous variables:

        (1) Calculate mean & variance for each continous variable.
        (2) Construct a gaussian with ^^.
        (3) Calculate entropy of each estimated gaussian.
    */

    val realOnly: D[DenseVector[Double]] =
      data.map { fv =>
        val realValues =
          realIndices
            .map { index =>
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

    val statsForAllRealFeatures =
      OnlineMeanVariance.batch[D, Double, DenseVector](realOnly)

    import MathOps.Implicits._
    val gf = GaussianFactory[Double]

    val realFeat2gaussian: Map[String, GaussianFactory[Double]#Gaussian] =
      realIndices
        .map { index =>
          val g = new gf.Gaussian(
            mean = statsForAllRealFeatures.mean(index),
            variance = statsForAllRealFeatures.variance(index),
            stddev = math.sqrt(statsForAllRealFeatures.variance(index))
          )
          (fs.features(index), g)
        }
        .toMap

    val realFeat2entropy: Map[String, Double] =
      realFeat2gaussian.map {
        case (real, gaussian) =>
          (real, entropyOf(gaussian))
      }

    //
    //
    //
    // DO CATEGORICAL VARIABLES
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

    val catFeat2Entropy: Map[String, Double] =
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

    //
    //
    //
    // PRODUCE ENTROPY FOR ALL FEATURES
    //
    //
    //

    fs.feat2index
      .map {
        case (featureName, index) =>
          if (catFeat2Entropy contains featureName)
            (catFeat2Entropy(featureName), index)
          else
            (realFeat2entropy(featureName), index)
      }
      .toSeq
      .sortBy { case (_, index) => index }
      .map { case (entropy, _) => entropy }
  }

  val logBase2 = logBaseX(2.0) _

  val logBaseE = logBaseX(math.E) _

  def logBaseX(base: Double)(value: Double): Double =
    math.log(value) / math.log(base)

  private[this] val gaussianEntropyConst = math.sqrt(2.0 * math.Pi * math.E)

  def entropyOf(g: GaussianFactory[Double]#Gaussian): Double =
    logBaseE(gaussianEntropyConst * g.stddev)

}

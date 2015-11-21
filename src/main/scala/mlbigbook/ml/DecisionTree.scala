package mlbigbook.ml

import breeze.linalg.{ Vector, DenseVector }
import fif.{ TravData, Data }
import mlbigbook.math.{ VectorOpsT, NumericConversion, OnlineMeanVariance }

import scala.annotation.tailrec
import scala.language.{ postfixOps, higherKinds }

object FeatureVectorSupport {

  sealed trait Value
  case class Categorical(v: String) extends Value
  case class Real(v: Double) extends Value

  type FeatVec = Seq[Value]

  case class FeatureSpace(
      features:           Seq[String],
      isCategorical:      Seq[Boolean],
      feat2index:         Map[String, Int],
      categorical2values: Map[String, Seq[String]]
  ) {

    val (realIndices, catIndices) =
      isCategorical
        .zipWithIndex
        .foldLeft((Seq.empty[Int], Seq.empty[Int])) {
          case ((ri, ci), (isCategory, index)) =>
            if (isCategory)
              (ri, ci :+ index)
            else
              (ri :+ index, ci)
        }

    val (realFeatNames, catFeatNames) = (
      realIndices.map { index => features(index) },
      catIndices.map { index => features(index) }
    )

    /*

       [PEDAGOGICAL NOTES for book writing]

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

  }

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

  def apply[D[_]: Data, T <: DecisionTree { type FeatureVector = Seq[String]; type Decision = Boolean }](
    dtModule: T,
    data:     D[(Seq[String], Boolean)]
  )(
    implicit
    fs: FeatureSpace
  ): Option[T#Node] = {
    implicit val _ = dtModule
    learn(data, 0 until fs.features.size)
  }

  protected def learn[D[_]: Data, T <: DecisionTree { type FeatureVector = Seq[String]; type Decision = Boolean }](
    data:         D[(Seq[String], Boolean)],
    featuresLeft: Seq[Int]
  )(
    implicit
    fs:       FeatureSpace,
    dtModule: T
  ): Option[T#Node] =

    if (data isEmpty)
      None

    else {

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
          Some(new dtModule.Leaf(true))
        else
          Some(new dtModule.Leaf(false))

      } else {

        (nPos, nNeg) match {

          case (0l, nonZero) =>
            Some(new dtModule.Leaf(false))

          case (nonZero, 0l) =>
            Some(new dtModule.Leaf(true))

          case (_, _) =>

            val entropyOfFeatures =
              InformationSimpleFv.entropyCategorical(
                data.map {
                  case (categoricalFeatures, _) => categoricalFeatures
                }
              )

            {
              implicit val v = TupleVal1[String]
              implicit val td = TravData
              Argmin(entropyOfFeatures.toTraversable)
            }
              .map {
                case (nameOfMinEntropyFeature, _) =>
                  // partition data according to the discrete values of each
                  val distinctValues = fs.categorical2values(nameOfMinEntropyFeature)

                  // how to partition ?
                  // IDEAS:
                  // (1) implement partition(f: A => Seq[B]): Map[B, D[A]] on Data type class
                  // (2) use map(f: A => Seq[B]) to turn into D[Seq[[(B, A])]]
                  //     then unroll with flatMap, getting D[(B,A)]
                  //     then filter according to each B, getting Seq[D[A]]

                  ???
              }
        }
      }
    }

}

object InformationSimpleFv {

  import fif.Data.ops._
  import FeatureVectorSupport._

  /*
    Strategy for continuous variables:

      (1) Calculate mean & variance for each continous variable.
      (2) Construct a gaussian with ^^.
      (3) Calculate entropy of each estimated gaussian.
   */
  def entropyContinous[D[_]: Data, N: NumericConversion, V[_] <: Vector[_]](
    realOnly: D[V[N]]
  )(
    implicit
    ops: VectorOpsT[N, V],
    fs:  FeatureSpace
  ): Map[String, Double] = {

    val statsForAllRealFeatures =
      OnlineMeanVariance.batch[D, N, V](realOnly)

    val gf = {
      import MathOps.Implicits._
      GaussianFactory[Double]
    }

    val realFeat2gaussian: Map[String, GaussianFactory[Double]#Gaussian] = {
      val toDouble = NumericConversion[N].numeric.toDouble _
      (0 until statsForAllRealFeatures.mean.size)
        .zip(fs.realFeatNames)
        .map {
          case (index, realFeatName) =>
            val g = new gf.Gaussian(
              mean = toDouble(ops.valueAt(statsForAllRealFeatures.mean)(index)),
              variance = toDouble(ops.valueAt(statsForAllRealFeatures.variance)(index)),
              stddev = toDouble(ops.valueAt(statsForAllRealFeatures.variance)(index))
            )
            (realFeatName, g)
        }
        .toMap
    }

    realFeat2gaussian
      .map {
        case (realFeatName, gaussian) =>
          (realFeatName, entropyOf(gaussian))
      }
  }

  /*
    Strategy for discrete variables:

    for each feature
      - count events

    for each feature
      for each event in feature:
        - calculate P(event) ==> # events / total
        - entropy(feature)   ==> - sum( p(event) * log_2( p(event) ) )
   */
  def entropyCategorical[D[_]: Data](
    categoricalOnly: D[Seq[String]]
  )(implicit fs: FeatureSpace): Map[String, Double] = {

    val catFeat2event2count =
      categoricalOnly
        .aggregate(Map.empty[String, Map[String, Long]])(
          {
            case (feat2event2count, featureValues) =>
              featureValues.zip(fs.catFeatNames)
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

    catFeat2event2count
      .map {
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
  }

  def entropy[D[_]: Data, FV](
    data: D[FV]
  )(
    implicit
    fs:   FeatureSpace,
    isFv: FV => FeatVec
  ): Seq[Double] = {

    val realFeat2entropy = {

      val realOnly: D[DenseVector[Double]] =
        data.map { fv =>
          val realValues =
            fs.realIndices
              .map { index =>
                fv(index) match {
                  case Real(v) => v
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
      entropyContinous(realOnly)
    }

    val catFeat2Entropy = {

      val categoricalOnly: D[Seq[String]] =
        data.map { fv =>
          fs.catIndices.map { index =>
            fv(index) match {
              case Categorical(v) => v
              case Real(_) =>
                throw new IllegalStateException(
                  s"Violation of FeatureSpace contract: feature at index $index is real, expecting categorical"
                )
            }
          }
        }

      entropyCategorical(categoricalOnly)
    }

    // put calculated entropy for both continuous and categorical features into
    // the same (feature name --> entropy) mapping
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

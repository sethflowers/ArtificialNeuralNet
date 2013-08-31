//-----------------------------------------------------------------------
// <copyright file="Program.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace DigitNet
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using ArtificialNeuralNet;
    using GeneticAlgorithm;

    /// <summary>
    /// A demonstration of using neural networks for digit classification (OCR).
    /// </summary>
    public class Program
    {
        /// <summary>
        /// The population size
        /// </summary>
        private const int PopulationSize = 200;

        /// <summary>
        /// The number of hidden nodes
        /// </summary>
        private const int NumberOfHiddenNodes = 300;

        /// <summary>
        /// The training size
        /// </summary>
        private const int TrainingSize = 500;

        /// <summary>
        /// The number of crossover points
        /// </summary>
        private const int NumberOfCrossoverPoints = 2;

        /// <summary>
        /// The crossover rate
        /// </summary>
        private const double CrossoverRate = 0.07;

        /// <summary>
        /// The mutation rate
        /// </summary>
        private const double MutationRate = 0.0075;

        /// <summary>
        /// The weight and bias range
        /// </summary>
        private const double WeightAndBiasRange = 2;

        /// <summary>
        /// The tournament percent
        /// </summary>
        private const double TournamentPercent = 0.2;

        /// <summary>
        /// The elitism percent
        /// </summary>
        private const double ElitismPercent = 0.2;

        /// <summary>
        /// The mutation type
        /// </summary>
        private const MutationStrategy MutationType = MutationStrategy.Random;

        /// <summary>
        /// The selection
        /// </summary>
        private static readonly string Selection = "Tournament";

        /// <summary>
        /// The load best ever
        /// </summary>
        private static readonly bool LoadBestEver = false;

        /// <summary>
        /// The main entry point into the demonstration.
        /// </summary>
        public static void Main()
        {
            List<TrainingData> trainingDataCollection = LoadTrainingData();

            Stopwatch watch = new Stopwatch();
            watch.Start();

            Run(trainingDataCollection);

            watch.Stop();
            Console.WriteLine("Elapsed Milliseconds : {0}", watch.ElapsedMilliseconds);

            Console.WriteLine("Done - hit enter to continue.");
            Console.ReadLine();
        }

        /// <summary>
        /// Runs a genetic algorithm trying to optimize the output of neural nets for classifying digits.
        /// </summary>
        /// <param name="trainingDataCollection">The collection of data with which to train the neural nets.</param>
        private static void Run(IList<TrainingData> trainingDataCollection)
        {
            int populationSize = PopulationSize;

            // The input layer needs a neuron for each pixel in a 28x28 image.
            // The output layer needs a neuron for each number from 0 to 9.
            int totalNeuronsInInputLayer = 28 * 28;
            int totalNeuronsInHiddenLayer = NumberOfHiddenNodes;
            int totalNeuronsInOutputLayer = 10;

            // Determine the total amount of data required for a single net with the given node counts.
            int totalInitializationDataPerNet = GetTotalInitializationDataPerNet(
                totalNeuronsInInputLayer, totalNeuronsInHiddenLayer, totalNeuronsInOutputLayer);

            int[] neuronsPerLayer = new[] 
            { 
                totalNeuronsInInputLayer, 
                totalNeuronsInHiddenLayer, 
                totalNeuronsInOutputLayer 
            };

            // Create the initial population of weights and biases for a collection of neural nets.
            ChromosomeCollection<double> beginningPopulation =
                GetBeginningPopulation(populationSize, totalInitializationDataPerNet, totalNeuronsInHiddenLayer);

            // Create a callback that allows us to provide a fitness for each chromosome.
            // Each chromosome represents the weights and biases for a single neural net.
            // The fitness is a function of how well the neural net can classify the digits in the sample data.
            ChromosomeFitnessCalculator<double> fitnessCalculator =
                new GenericFitnessCalculator<double>(chromosome =>
                    GetFitnessForNet(trainingDataCollection, chromosome, neuronsPerLayer));

            // Create a population evolver for the genetic algorithm.
            Random random = new Random();
            PopulationEvolver<double> populationEvolver =
                new PopulationEvolver<double>(
                    selector: CreateChromosomeSelector(random, populationSize),
                    modifier: CreateChromosomeModifier(random),
                    validator: new GenericChromosomeValidator<double>(c => true));

            // Create a genetic algorithm.
            GA<double> ga = new GA<double>(fitnessCalculator, populationEvolver);

            int currentGeneration = 0;

            // Output the best fitness every epoch.
            ga.Epoch += (o, e) =>
            {
                Chromosome<double> bestChromosome = e.Data.OrderByDescending(c => c.Fitness).First();

                Console.WriteLine(
                    "Gen: {0}, Best: {1}, Correct: {2}, Avg: {3}, Time: {4}",
                    ++currentGeneration,
                    Math.Round(e.Data.Max(c => c.Fitness), 5),
                    GetTotalCorrect(trainingDataCollection, neuronsPerLayer, bestChromosome),
                    Math.Round(e.Data.Average(c => c.Fitness), 5),
                    DateTime.Now.ToString("hh:mm:ss.fff"));

                if (bestChromosome.Fitness > 99)
                {
                    string fileName = string.Format("chromosomes.{0}hidden.99plusfitness.txt", totalNeuronsInHiddenLayer);

                    File.WriteAllText(
                        path: fileName,
                        contents: string.Join(",", bestChromosome.Genes) + Environment.NewLine);
                }

                string file = string.Format("chromosomes.{0}hidden.txt", totalNeuronsInHiddenLayer);

                File.WriteAllText(
                    path: file,
                    contents: string.Join(",", bestChromosome.Genes) + Environment.NewLine);
            };

            // Run the genetic algorithm.
            ChromosomeCollection<double> endingPopulation =
                ga.Run(beginningPopulation, numberOfGenerations: 1000, numberOfElites: (int)(populationSize * ElitismPercent));

            Console.WriteLine("here");
        }

        /// <summary>
        /// Creates the chromosome selector.
        /// </summary>
        /// <param name="random">The random.</param>
        /// <param name="populationSize">Size of the population.</param>
        /// <returns>Returns a chromosome selector.</returns>
        private static ChromosomeSelector<double> CreateChromosomeSelector(Random random, int populationSize)
        {
            if (Selection == "Tournament")
            {
                return new TournamentSelector<double>(
                    random, numberOfPlayers: (int)(populationSize * TournamentPercent));
            }

            return new RouletteSelector<double>(random);
        }

        /// <summary>
        /// Creates a chromosome modifier.
        /// </summary>
        /// <param name="random">The random number generator.</param>
        /// <returns>Returns a chromosome modifier.</returns>
        private static ChromosomeModifier<double> CreateChromosomeModifier(Random random)
        {
            return
                new ChromosomeModifier<double>(
                    random,
                    mutationRate: MutationRate,
                    crossoverRate: CrossoverRate,
                    numberOfCrossoverPoints: NumberOfCrossoverPoints,
                    mutationStrategy: MutationType,
                    mutationFunction: null);
        }

        /// <summary>
        /// Gets the total count of initialization data required for a single neural net.
        /// </summary>
        /// <param name="totalNeuronsInInputLayer">The total amount of neurons in the input layer.</param>
        /// <param name="totalNeuronsInHiddenLayer">The total amount of neurons in the hidden layer.</param>
        /// <param name="totalNeuronsInOutputLayer">The total amount of neurons in the output layer.</param>
        /// <returns>Returns the total count of initialization data required for a single neural net.</returns>
        private static int GetTotalInitializationDataPerNet(
            int totalNeuronsInInputLayer,
            int totalNeuronsInHiddenLayer,
            int totalNeuronsInOutputLayer)
        {
            // Each node in each layer other than the input layer needs 
            // a weight for each node in the previous layer.
            // The input layer does not require inputs.
            int totalWeights =
                (totalNeuronsInHiddenLayer * totalNeuronsInInputLayer) +
                (totalNeuronsInOutputLayer * totalNeuronsInHiddenLayer);

            // Each node in each layer other than the input layer needs a single bias.
            // The input layer does not require biases.
            int totalBiases =
                totalNeuronsInHiddenLayer +
                totalNeuronsInOutputLayer;

            return totalWeights + totalBiases;
        }

        /// <summary>
        ///   <para>Gets the beginning population of data to run through a genetic algorithm.</para>
        ///   <para>Each chromosome in the population will represent the data for a single neural net.</para>
        ///   <para>The genes in each chromosome will be the weights and biases used by the neurons in the net.</para>
        /// </summary>
        /// <param name="populationSize">The size of the population of chromosomes to create.</param>
        /// <param name="totalInitializationDataPerNet">The total count of initialization data for each neural net.</param>
        /// <param name="totalNeuronsInHiddenLayer">The total number of neurons in the hidden layer.</param>
        /// <returns>
        /// Returns the beginning population of data to run through a genetic algorithm.
        /// </returns>
        private static ChromosomeCollection<double> GetBeginningPopulation(
            int populationSize,
            int totalInitializationDataPerNet,
            int totalNeuronsInHiddenLayer)
        {
            ChromosomeCollection<double> beginningPopulation = new ChromosomeCollection<double>();
            Random random = new Random();

            if (LoadBestEver)
            {
                string file = string.Format("chromosomes.{0}hidden.txt", totalNeuronsInHiddenLayer);

                if (File.Exists(file))
                {
                    foreach (string line in File.ReadAllLines(file))
                    {
                        Chromosome<double> chromosome = new Chromosome<double>(
                            genes: line.Split(',').Select(s => double.Parse(s)).ToList());
                        beginningPopulation.Add(chromosome);
                    }
                }
            }

            while (beginningPopulation.Count < populationSize)
            {
                List<double> initializationData = new List<double>(
                    capacity: totalInitializationDataPerNet);

                for (int j = 0; j < totalInitializationDataPerNet; j++)
                {
                    initializationData.Add((random.NextDouble() * WeightAndBiasRange) - (WeightAndBiasRange / 2));
                }

                beginningPopulation.Add(new Chromosome<double>(initializationData));
            }

            return beginningPopulation;
        }

        /// <summary>
        /// Gets the fitness for a single neural net whose weights and biases are represented by the genes in the given chromosome.
        /// </summary>
        /// <param name="trainingDataCollection">The data used to train the nets.</param>
        /// <param name="chromosome">The chromosome whose genes are used to initialize the neural net.</param>
        /// <param name="neuronsPerLayer">The neurons per layer in the neural net.</param>
        /// <returns>Returns the fitness for a single neural net whose weights and biases are represented by the genes in the given chromosome.</returns>
        private static double GetFitnessForNet(
            IList<TrainingData> trainingDataCollection,
            Chromosome<double> chromosome,
            int[] neuronsPerLayer)
        {
            ////int[] neuronsPerLayerAfterInputLayer = neuronsPerLayer.Skip(1).ToArray();
            ////int numberOfBiases = neuronsPerLayerAfterInputLayer.Sum();
            ////double[] biases = chromosome.Genes.Take(numberOfBiases).ToArray();
            ////double[] weights = chromosome.Genes.Skip(numberOfBiases).ToArray();

            ////double correctOutput = 0;
            ////double incorrectOutput = 0;

            ////foreach (TrainingData trainingData in trainingDataCollection)
            ////{
            ////    double[] output = NeuralNet.ThinkFast(
            ////        inputs: trainingData.Data,
            ////        neuronsPerLayerAfterInputLayer: neuronsPerLayerAfterInputLayer,
            ////        biases: biases,
            ////        weights: weights);

            ////    for (int i = 0; i < output.Length; i++)
            ////    {
            ////        if (i == trainingData.Digit)
            ////        {
            ////            correctOutput += output[i];
            ////        }
            ////        else
            ////        {
            ////            incorrectOutput += output[i];
            ////        }
            ////    }
            ////}

            ////return Math.Round(correctOutput / incorrectOutput, 5);

            int totalCorrect = 0;

            int[] neuronsPerLayerAfterInputLayer = neuronsPerLayer.Skip(1).ToArray();
            int numberOfBiases = neuronsPerLayerAfterInputLayer.Sum();
            double[] biases = chromosome.Genes.Take(numberOfBiases).ToArray();
            double[] weights = chromosome.Genes.Skip(numberOfBiases).ToArray();

            Parallel.ForEach(
                trainingDataCollection, 
                trainingData =>
                {
                    double[] output = NeuralNet.ThinkFast(
                        inputs: trainingData.Data,
                        neuronsPerLayerAfterInputLayer: neuronsPerLayerAfterInputLayer,
                        biases: biases,
                        weights: weights);

                    double highestGuess = double.MinValue;
                    int guess = int.MinValue;

                    for (int i = 0; i < output.Length; i++)
                    {
                        if (output[i] > highestGuess)
                        {
                            highestGuess = output[i];
                            guess = i;
                        }
                    }

                    if (guess == trainingData.Digit)
                    {
                        totalCorrect++;
                    }
                });

            return Math.Round(((double)totalCorrect / trainingDataCollection.Count) * 100, 5);
        }

        /// <summary>
        /// Gets the total digits classified correctly from the training set by the neural net initialized with the genes in the given chromosome.
        /// </summary>
        /// <param name="trainingDataCollection">The training data collection.</param>
        /// <param name="neuronsPerLayer">The neurons per layer.</param>
        /// <param name="chromosome">The chromosome.</param>
        /// <returns>Returns the total digits classified correctly from the training set by the neural net initialized with the genes in the given chromosome.</returns>
        private static int GetTotalCorrect(
            IList<TrainingData> trainingDataCollection,
            int[] neuronsPerLayer,
            Chromosome<double> chromosome)
        {
            int totalCorrect = 0;

            int[] neuronsPerLayerAfterInputLayer = neuronsPerLayer.Skip(1).ToArray();
            int numberOfBiases = neuronsPerLayerAfterInputLayer.Sum();
            double[] biases = chromosome.Genes.Take(numberOfBiases).ToArray();
            double[] weights = chromosome.Genes.Skip(numberOfBiases).ToArray();

            foreach (TrainingData trainingData in trainingDataCollection)
            {
                double[] output = NeuralNet.ThinkFast(
                    inputs: trainingData.Data,
                    neuronsPerLayerAfterInputLayer: neuronsPerLayerAfterInputLayer,
                    biases: biases,
                    weights: weights);

                double highestGuess = double.MinValue;
                int guess = int.MinValue;

                for (int i = 0; i < output.Length; i++)
                {
                    if (output[i] > highestGuess)
                    {
                        highestGuess = output[i];
                        guess = i;
                    }
                }

                if (guess == trainingData.Digit)
                {
                    totalCorrect++;
                }
            }

            return totalCorrect;

            ////NeuralNet neuralNet = new NeuralNet(
            ////    neuronsPerLayer: neuronsPerLayer,
            ////    weightsAndBiases: chromosome.Genes.ToArray());

            ////int totalCorrect = 0;

            ////foreach (TrainingData trainingData in trainingDataCollection)
            ////{
            ////    IList<double> output = neuralNet.Think(trainingData.Data.ToList()).ToList();

            ////    double highestGuess = double.MinValue;
            ////    int guess = int.MinValue;

            ////    for (int i = 0; i < output.Count; i++)
            ////    {
            ////        if (output[i] > highestGuess)
            ////        {
            ////            highestGuess = output[i];
            ////            guess = i;
            ////        }
            ////    }

            ////    if (guess == trainingData.Digit)
            ////    {
            ////        totalCorrect++;
            ////    }
            ////}

            ////return totalCorrect;
        }

        /// <summary>
        /// Loads all the training data from a file.
        /// </summary>
        /// <returns>Returns a collection of the training data.</returns>
        private static List<TrainingData> LoadTrainingData()
        {
            List<TrainingData> trainingDataCollection = new List<TrainingData>();

            bool skipLine = true;

            int trainingDataSize = TrainingSize;

            foreach (string data in File.ReadLines("../../Files/train.csv"))
            {
                if (skipLine)
                {
                    skipLine = false;
                    continue;
                }
                else if (trainingDataSize-- == 0)
                {
                    break;
                }

                TrainingData trainingData = new TrainingData();
                trainingData.Digit = (short)(data[0] - (int)'0');

                string[] lineData = data.Split(",".ToCharArray());
                ////trainingData.Data = lineData.Skip(1).Select(s => double.Parse(s) > 0 ? 1d : 0).ToArray();
                trainingData.Data = lineData.Skip(1).Select(s => double.Parse(s) / byte.MaxValue).ToArray();

                trainingDataCollection.Add(trainingData);
            }

            return trainingDataCollection;
        }
    }
}

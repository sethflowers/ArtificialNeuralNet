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
    using ArtificialNeuralNet;
    using GeneticAlgorithm;

    /// <summary>
    /// A demonstration of using neural networks for digit classification (OCR).
    /// </summary>
    public class Program
    {
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
            int populationSize = 100;

            // The input layer needs a neuron for each pixel in a 28x28 image.
            // The output layer needs a neuron for each number from 0 to 9.
            int totalNeuronsInInputLayer = 28 * 28;
            int totalNeuronsInHiddenLayer = 50;
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
                GetBeginningPopulation(populationSize, totalInitializationDataPerNet);

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
                    selector: new RouletteSelector<double>(random),
                    modifier: new ChromosomeModifier<double>(random),
                    validator: new GenericChromosomeValidator<double>(c => true));
            
            // Create a genetic algorithm.
            GA<double> ga = new GA<double>(fitnessCalculator, populationEvolver);

            int currentGeneration = 0;

            // Output the best fitness every epoch.
            ga.Epoch += (o, e) =>
            {
                Chromosome<double> bestChromosome = e.Data.OrderByDescending(c => c.Fitness).First();

                Console.WriteLine(
                    "Gen: {0}, Best: {1}, Avg: {2}, Correct: {3}, Time: {4}", 
                    ++currentGeneration,
                    Math.Round(e.Data.Max(c => c.Fitness), 5),
                    Math.Round(e.Data.Average(c => c.Fitness), 5),
                    GetTotalCorrect(trainingDataCollection, neuronsPerLayer, bestChromosome),
                    DateTime.Now.ToString("hh:mm:ss.fff"));
            };

            // Run the genetic algorithm.
            ChromosomeCollection<double> endingPopulation =
                ga.Run(beginningPopulation, numberOfGenerations: 1, numberOfBestChromosomesToPromote: populationSize / 5);

            Console.WriteLine("here");
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
        /// <para>Gets the beginning population of data to run through a genetic algorithm.</para>
        /// <para>Each chromosome in the population will represent the data for a single neural net.</para>
        /// <para>The genes in each chromosome will be the weights and biases used by the neurons in the net.</para>
        /// </summary>
        /// <param name="populationSize">The size of the population of chromosomes to create.</param>
        /// <param name="totalInitializationDataPerNet">The total count of initialization data for each neural net.</param>
        /// <returns>Returns the beginning population of data to run through a genetic algorithm.</returns>
        private static ChromosomeCollection<double> GetBeginningPopulation(
            int populationSize,
            int totalInitializationDataPerNet)
        {
            ChromosomeCollection<double> beginningPopulation = new ChromosomeCollection<double>();
            Random random = new Random();

            for (int i = 0; i < populationSize; i++)
            {
                List<double> initializationData = new List<double>(
                    capacity: totalInitializationDataPerNet);

                for (int j = 0; j < totalInitializationDataPerNet; j++)
                {
                    initializationData.Add((random.NextDouble() * 10) - 5);
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
            return GetTotalCorrect(trainingDataCollection, neuronsPerLayer, chromosome);

            ////NeuralNet neuralNet = new NeuralNet(
            ////    neuronsPerLayer: neuronsPerLayer,
            ////    weightsAndBiases: chromosome.Genes.ToArray());

            ////double correctTotal = 0;
            ////double incorrectTotal = 0;

            ////foreach (TrainingData trainingData in trainingDataCollection)
            ////{
            ////    IList<double> output = neuralNet.Think(trainingData.Data.ToList()).ToList();

            ////    for (int i = 0; i < output.Count; i++)
            ////    {
            ////        if (i == trainingData.Digit)
            ////        {
            ////            correctTotal += output[i];
            ////        }
            ////        else
            ////        {
            ////            incorrectTotal += output[i];
            ////        }
            ////    }
            ////}

            ////return correctTotal / incorrectTotal;
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
            NeuralNet neuralNet = new NeuralNet(
                neuronsPerLayer: neuronsPerLayer,
                weightsAndBiases: chromosome.Genes.ToArray());

            int totalCorrect = 0;

            foreach (TrainingData trainingData in trainingDataCollection)
            {
                IList<double> output = neuralNet.Think(trainingData.Data.ToList()).ToList();

                double highestGuess = double.MinValue;
                int guess = int.MinValue;

                for (int i = 0; i < output.Count; i++)
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
        }

        /// <summary>
        /// Loads all the training data from a file.
        /// </summary>
        /// <returns>Returns a collection of the training data.</returns>
        private static List<TrainingData> LoadTrainingData()
        {
            List<TrainingData> trainingDataCollection = new List<TrainingData>();

            bool skipLine = true;

            int trainingDataSize = 500;

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
                trainingData.Data = lineData.Skip(1).Select(s => double.Parse(s) / byte.MaxValue).ToArray();

                trainingDataCollection.Add(trainingData);
            }

            return trainingDataCollection;
        }
    }
}

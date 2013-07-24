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
                GetBeginningPopulation(
                    populationSize: 100,
                    totalInitializationDataPerNet: totalInitializationDataPerNet);

            // Create a callback that allows us to provide a fitness for each chromosome.
            // Each chromosome represents the weights and biases for a single neural net.
            // The fitness is a function of how well the neural net can classify the digits in the sample data.
            ChromosomeFitnessCalculator<double> fitnessCalculator =
                new GenericFitnessCalculator<double>(chromosome =>
                    GetFitnessForNet(trainingDataCollection, chromosome, neuronsPerLayer));

            // Run the genetic algorithm.
            ChromosomeCollection<double> endingPopulation =
                new GA<double>(fitnessCalculator)
                    .Run(beginningPopulation, numberOfGenerations: 10);

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
            // Each node in each layer needs a weight for each node in the previous layer.
            // The input layer has a single weight for each node.
            int totalWeights = totalNeuronsInInputLayer +
                (totalNeuronsInHiddenLayer * totalNeuronsInInputLayer) +
                (totalNeuronsInOutputLayer * totalNeuronsInHiddenLayer);

            // Each node in each layer needs a single bias.
            int totalBiases = totalNeuronsInInputLayer +
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
                    initializationData.Add((random.NextDouble() * 2) - 1);
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
            NeuralNet neuralNet = new NeuralNet(
                neuronsPerLayer: neuronsPerLayer,
                weightsAndBiases: chromosome.Genes.ToArray());

            double correctTotal = 0;
            double incorrectTotal = 0;

            foreach (TrainingData trainingData in trainingDataCollection.Take(500))
            {
                IList<double> dataAsDoubles = trainingData.Data.Select(b => (double)b / byte.MaxValue).ToList();
                IList<double> output = neuralNet.Think(dataAsDoubles).ToList();

                for (int i = 0; i < output.Count; i++)
                {
                    if (i == trainingData.Digit)
                    {
                        correctTotal += output[i];
                    }
                    else
                    {
                        incorrectTotal += output[i];
                    }
                }
            }

            return correctTotal / incorrectTotal;
        }

        /// <summary>
        /// Loads all the training data from a file.
        /// </summary>
        /// <returns>Returns a collection of the training data.</returns>
        private static List<TrainingData> LoadTrainingData()
        {
            List<TrainingData> trainingDataCollection = new List<TrainingData>();

            bool skipLine = true;

            foreach (string data in File.ReadAllLines("../../Files/train.csv"))
            {
                if (skipLine)
                {
                    skipLine = false;
                    continue;
                }

                TrainingData trainingData = new TrainingData();
                trainingData.Digit = (short)(data[0] - (int)'0');

                string[] lineData = data.Split(",".ToCharArray());
                trainingData.Data = lineData.Skip(1).Select(s => byte.Parse(s)).ToArray();

                trainingDataCollection.Add(trainingData);
            }

            return trainingDataCollection;
        }
    }
}

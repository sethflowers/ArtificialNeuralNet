//-----------------------------------------------------------------------
// <copyright file="Program.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace DigitNet
{
    using System;
    using System.Collections.Generic;
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

            ////ChromosomeFitnessCalculator<double> fitnessCalculator =
            ////    new GenericFitnessCalculator<double>(chromosome =>
            ////        {

            ////        });

            ////GA<double> ga = new GA<double>(fitnessCalculator);
            ////ChromosomeCollection<double> beginningPopulation = new ChromosomeCollection<double>();
            ////for (int i = 0; i < 100; i++)
            ////{
            ////    beginningPopulation.Add(new Chromosome<double>());
            ////}

            // There are 784 inputs, because the image is 28x28 pixels.
            NeuralNet neuralNet = new NeuralNet(new[] { 784, 50, 10 });

            int amountCorrect = 0;

            foreach (TrainingData trainingData in trainingDataCollection.Take(1000))
            {
                IList<double> dataAsDoubles = trainingData.Data.Select(b => (double)b / byte.MaxValue).ToList();
                IList<double> output = neuralNet.Think(dataAsDoubles).ToList();

                double currentMax = double.MinValue;
                int actualValue = int.MinValue;

                for (int i = 0; i < output.Count; i++)
                {
                    if (output[i] > currentMax)
                    {
                        currentMax = output[i];
                        actualValue = i;
                    }
                }

                amountCorrect += trainingData.Digit == actualValue ? 1 : 0;

            ////    Console.WriteLine("{0}, Expected: {1}, Actual: {2}", trainingData.Digit == actualValue ? "Success" : "Fail", trainingData.Digit, actualValue);
            }
            
            Console.WriteLine("Amount correct: {0}", amountCorrect);
            Console.WriteLine("Done - hit enter to continue.");
            Console.ReadLine();
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

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

            Console.WriteLine(trainingDataCollection.Count);
            Console.WriteLine(trainingDataCollection[0].Digit);

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

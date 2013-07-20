//-----------------------------------------------------------------------
// <copyright file="Synapse.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace ArtificialNeuralNet
{
    using System;

    /// <summary>
    /// A connection between neurons, providing input and output.
    /// </summary>
    public class Synapse
    {
        /// <summary>
        /// Provides a mechanism to randomize the weight of the synaptic connection.
        /// </summary>
        private static readonly Random Randomizer = new Random();

        /// <summary>
        /// Initializes a new instance of the <see cref="Synapse"/> class.
        /// </summary>
        public Synapse()
        {
            // Randomize the weight between -1 and 1.
            this.Weight = (Randomizer.NextDouble() * 2) - 1;
        }
        
        /// <summary>
        /// Gets or sets the weight of the connection.
        /// </summary>
        /// <value>
        /// The weight of the connection.
        /// </value>
        public double Weight { get; set; }

        /// <summary>
        /// Gets or sets the value passing through the connection.
        /// </summary>
        /// <value>
        /// The value passing through the connection.
        /// </value>
        public double Value { get; set; }
    }
}

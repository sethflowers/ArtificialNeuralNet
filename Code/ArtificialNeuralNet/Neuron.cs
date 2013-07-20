//-----------------------------------------------------------------------
// <copyright file="Neuron.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace ArtificialNeuralNet
{
    using System;
    using System.Collections.ObjectModel;

    /// <summary>
    /// Represents a neuron in the brain, connected to other neurons via inputs and outputs.
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// Provides a mechanism to randomize the bias.
        /// </summary>
        private static readonly Random Randomizer = new Random();

        /// <summary>
        /// Initializes a new instance of the <see cref="Neuron" /> class.
        /// </summary>
        public Neuron()
        {
            this.Inputs = new Collection<Synapse>();
            this.Outputs = new Collection<Synapse>();
            
            // Randomize the bias between -1 and 1.
            this.Bias = (Randomizer.NextDouble() * 2) - 1;
        }

        /// <summary>
        /// Gets the weights of the synapses that provide input to this neuron.
        /// </summary>
        /// <value>
        /// The neurons that provide input to this neuron.
        /// </value>
        public Collection<Synapse> Inputs { get; private set; }

        /// <summary>
        /// Gets the weights of the synapses that provide output to this neuron.
        /// </summary>
        /// <value>
        /// The neurons that provide output to this neuron.
        /// </value>
        public Collection<Synapse> Outputs { get; private set; }

        /// <summary>
        /// <para>Gets or sets the bias weight to the neuron.</para>
        /// <para>The bias allows the output of the neuron to be shifted (translated).</para>
        /// </summary>
        /// <value>
        /// The bias weight.
        /// </value>
        public double Bias { get; set; }

        /// <summary>
        /// Runs the activation function on the inputs to this neuron, setting the output value on each output.
        /// </summary>
        public virtual void Think()
        {
            double sum = this.Bias;

            foreach (Synapse input in this.Inputs)
            {
                sum += input.Value * input.Weight;
            }

            // Run the total through a sigmoid function.
            // If this calculation is causing poor performance, we can approximate it with the following.
            // double value = ((0.5 * sum) / (Math.Abs(sum) + 1)) + 0.5;
            double value = 1 / (1 + Math.Pow(Math.E, -sum));

            foreach (Synapse output in this.Outputs)
            {
                output.Value = value;
            }
        }
    }
}
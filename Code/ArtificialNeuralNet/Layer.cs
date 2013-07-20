//-----------------------------------------------------------------------
// <copyright file="Layer.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace ArtificialNeuralNet
{
    using System;
    using System.Collections.ObjectModel;

    /// <summary>
    /// Represents a layer of neurons in the artificial neural net,
    /// where each neuron in the layer can be connected to neurons in other layers.
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Layer"/> class.
        /// </summary>
        public Layer()
            : this(numberOfNeurons: 1)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Layer" /> class.
        /// </summary>
        /// <param name="numberOfNeurons">The number of neurons in the layer.</param>
        /// <exception cref="System.ArgumentException">A neural layer cannot have a negative number of neurons.;numberOfNeurons</exception>
        public Layer(int numberOfNeurons)
        {
            if (numberOfNeurons < 1)
            {
                throw new ArgumentException("A neural layer cannot have a negative number of neurons.", "numberOfNeurons");
            }

            // Initialize the collection of neurons.
            this.Neurons = new Collection<Neuron>();

            // Add the required neurons to this layer.
            while (numberOfNeurons-- > 0)
            {
                this.Neurons.Add(new Neuron());
            }
        }

        /// <summary>
        /// Gets the neurons in this layer.
        /// </summary>
        /// <value>
        /// The neurons in this layer.
        /// </value>
        public Collection<Neuron> Neurons { get; private set; }

        /// <summary>
        /// Causes each neuron in this layer to process its input and produce an output.
        /// </summary>
        public void Think()
        {
            foreach (Neuron neuron in this.Neurons)
            {
                neuron.Think();
            }
        }
    }
}
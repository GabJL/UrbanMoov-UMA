package es.uma.auxiliar;

// Copyright (c) 2007-2020 VMware, Inc. or its affiliates.  All rights reserved.
//
// This software, the RabbitMQ Java client library, is triple-licensed under the
// Mozilla Public License 2.0 ("MPL"), the GNU General Public License version 2
// ("GPL") and the Apache License version 2 ("ASL"). For the MPL, please see
// LICENSE-MPL-RabbitMQ. For the GPL, please see LICENSE-GPL2.  For the ASL,
// please see LICENSE-APACHE2.
//
// This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY KIND,
// either express or implied. See the LICENSE file for specific language governing
// rights and limitations of this software.
//
// If you have any questions regarding licensing, please contact us at
// info@rabbitmq.com.


import java.io.IOException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import com.rabbitmq.client.*;
import com.rabbitmq.client.AMQP.BasicProperties;
import com.rabbitmq.utility.Utility;

public class QueueingConsumer extends DefaultConsumer {
    private final BlockingQueue<Delivery> _queue;

    // When this is non-null the queue is in shutdown mode and nextDelivery should
    // throw a shutdown signal exception.
    private volatile ShutdownSignalException _shutdown;
    private volatile ConsumerCancelledException _cancelled;

    // Marker object used to signal the queue is in shutdown mode.
    // It is only there to wake up consumers. The canonical representation
    // of shutting down is the presence of _shutdown.
    // Invariant: This is never on _queue unless _shutdown != null.
    private static final Delivery POISON = new Delivery(null, null, null);

    public QueueingConsumer(Channel ch) {
        this(ch, new LinkedBlockingQueue<Delivery>());
    }

    public QueueingConsumer(Channel ch, BlockingQueue<Delivery> q) {
        super(ch);
        this._queue = q;
    }

    @Override public void handleShutdownSignal(String consumerTag,
                                               ShutdownSignalException sig) {
        _shutdown = sig;
        _queue.add(POISON);
    }

    @Override public void handleCancel(String consumerTag) throws IOException {
        _cancelled = new ConsumerCancelledException();
        _queue.add(POISON);
    }

    @Override public void handleDelivery(String consumerTag,
                                         Envelope envelope,
                                         AMQP.BasicProperties properties,
                                         byte[] body)
            throws IOException
    {
        checkShutdown();
        this._queue.add(new Delivery(envelope, properties, body));
    }

    /**
     * Encapsulates an arbitrary message - simple "bean" holder structure.
     */
    public static class Delivery {
        private final Envelope _envelope;
        private final AMQP.BasicProperties _properties;
        private final byte[] _body;

        public Delivery(Envelope envelope, AMQP.BasicProperties properties, byte[] body) {
            _envelope = envelope;
            _properties = properties;
            _body = body;
        }

        /**
         * Retrieve the message envelope.
         * @return the message envelope
         */
        public Envelope getEnvelope() {
            return _envelope;
        }

        /**
         * Retrieve the message properties.
         * @return the message properties
         */
        public BasicProperties getProperties() {
            return _properties;
        }

        /**
         * Retrieve the message body.
         * @return the message body
         */
        public byte[] getBody() {
            return _body;
        }
    }

    /**
     * Check if we are in shutdown mode and if so throw an exception.
     */
    private void checkShutdown() {
        if (_shutdown != null)
            throw Utility.fixStackTrace(_shutdown);
    }

    /**
     * If delivery is not POISON nor null, return it.
     * <p/>
     * If delivery, _shutdown and _cancelled are all null, return null.
     * <p/>
     * If delivery is POISON re-insert POISON into the queue and
     * throw an exception if POISONed for no reason.
     * <p/>
     * Otherwise, if we are in shutdown mode or cancelled,
     * throw a corresponding exception.
     */
    private Delivery handle(Delivery delivery) {
        if (delivery == POISON ||
                delivery == null && (_shutdown != null || _cancelled != null)) {
            if (delivery == POISON) {
                _queue.add(POISON);
                if (_shutdown == null && _cancelled == null) {
                    throw new IllegalStateException(
                            "POISON in queue, but null _shutdown and null _cancelled. " +
                                    "This should never happen, please report as a BUG");
                }
            }
            if (null != _shutdown)
                throw Utility.fixStackTrace(_shutdown);
            if (null != _cancelled)
                throw Utility.fixStackTrace(_cancelled);
        }
        return delivery;
    }

    /**
     * Main application-side API: wait for the next message delivery and return it.
     * @return the next message
     * @throws InterruptedException if an interrupt is received while waiting
     * @throws ShutdownSignalException if the connection is shut down while waiting
     * @throws ConsumerCancelledException if this consumer is cancelled while waiting
     */
    public Delivery nextDelivery()
            throws InterruptedException, ShutdownSignalException, ConsumerCancelledException
    {
        return handle(_queue.take());
    }

    /**
     * Main application-side API: wait for the next message delivery and return it.
     * @param timeout timeout in millisecond
     * @return the next message or null if timed out
     * @throws InterruptedException if an interrupt is received while waiting
     * @throws ShutdownSignalException if the connection is shut down while waiting
     * @throws ConsumerCancelledException if this consumer is cancelled while waiting
     */
    public Delivery nextDelivery(long timeout)
            throws InterruptedException, ShutdownSignalException, ConsumerCancelledException
    {
        return handle(_queue.poll(timeout, TimeUnit.MILLISECONDS));
    }
}
